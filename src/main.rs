use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{Stream, StreamConfig};
use rustfft::{num_complex::Complex, FftPlanner};
use std::collections::VecDeque;
use std::io::{self, Write};
use std::sync::{Arc, Mutex};

// ============================================================
// 处理模式配置
// ============================================================

#[derive(Clone, Copy, Debug)]
enum ProcessingMode {
    Performance,
    Balanced,
    Deep,
}

impl ProcessingMode {
    fn frame_duration_ms(&self) -> f32 {
        match self {
            Self::Performance => 16.0,
            Self::Balanced => 32.0,
            Self::Deep => 46.0,
        }
    }

    fn overlap_ratio(&self) -> f32 {
        match self {
            Self::Performance => 0.0,
            Self::Balanced => 0.5,
            Self::Deep => 0.875,
        }
    }

    fn similarity_threshold(&self) -> f32 {
        match self {
            Self::Performance => 0.70,
            Self::Balanced => 0.65,
            Self::Deep => 0.55,
        }
    }

    fn suppress_gain(&self) -> f32 {
        match self {
            Self::Performance => 0.0,
            Self::Balanced => 0.0,
            Self::Deep => 0.0,
        }
    }

    fn enhance_gain(&self) -> f32 {
        match self {
            Self::Performance => 2.5,
            Self::Balanced => 2.0,
            Self::Deep => 3.0,
        }
    }

    fn smoothing_coeff(&self) -> f32 {
        match self {
            Self::Performance => 0.70,
            Self::Balanced => 0.60,
            Self::Deep => 0.80,
        }
    }

    fn temporal_smooth_frames(&self) -> usize {
        match self {
            Self::Performance => 1,
            Self::Balanced => 3,
            Self::Deep => 5,
        }
    }

    fn max_targets(&self) -> usize {
        match self {
            Self::Performance => 1,
            Self::Balanced => usize::MAX,
            Self::Deep => usize::MAX,
        }
    }

    fn uses_fft(&self) -> bool {
        !matches!(self, Self::Performance)
    }

    fn uses_window(&self) -> bool {
        matches!(self, Self::Deep)
    }

    fn uses_overlap(&self) -> bool {
        !matches!(self, Self::Performance)
    }
}

// ============================================================
// 目标人声
// ============================================================

#[derive(Clone, Copy, Debug, PartialEq)]
enum TargetAction {
    Suppress,
    Enhance,
}

struct TargetVoice {
    name: String,
    action: TargetAction,
    reference_features: Vec<Vec<f32>>,
}

// ============================================================
// 特征提取
// ============================================================

/// 4 子带能量比特征（性能模式）
fn extract_subband_energy(frame: &[f32]) -> Vec<f32> {
    let n = frame.len();
    let band1_end = n / 4;
    let band2_end = n / 2;
    let band3_end = 3 * n / 4;

    let e1: f32 = frame[..band1_end].iter().map(|&x| x * x).sum();
    let e2: f32 = frame[band1_end..band2_end].iter().map(|&x| x * x).sum();
    let e3: f32 = frame[band2_end..band3_end].iter().map(|&x| x * x).sum();
    let e4: f32 = frame[band3_end..].iter().map(|&x| x * x).sum();

    let total = e1 + e2 + e3 + e4;
    if total < 1e-10 {
        return vec![0.25, 0.25, 0.25, 0.25];
    }
    vec![e1 / total, e2 / total, e3 / total, e4 / total]
}

/// FFT 幅度谱特征 → 梅尔频段降维（平衡/深度模式）
fn extract_mel_spectrum(frame: &[f32], planner: &mut FftPlanner<f32>, sample_rate: u32, n_mels: usize) -> Vec<f32> {
    let n = frame.len();
    let fft = planner.plan_fft_forward(n);
    let mut buffer: Vec<Complex<f32>> = frame
        .iter()
        .map(|&x| Complex::new(x, 0.0))
        .collect();
    fft.process(&mut buffer);

    let spectrum_len = n / 2 + 1;
    let power: Vec<f32> = buffer[..spectrum_len]
        .iter()
        .map(|c| c.norm_sqr() / (n as f32))
        .collect();

    // 梅尔频段滤波器组
    let mel_low = hz_to_mel(80.0);
    let mel_high = hz_to_mel(sample_rate as f32 / 2.0);
    let mel_points: Vec<f32> = (0..=n_mels + 1)
        .map(|i| mel_low + (mel_high - mel_low) * i as f32 / (n_mels + 1) as f32)
        .collect();
    let hz_points: Vec<f32> = mel_points.iter().map(|&m| mel_to_hz(m)).collect();
    let bin_points: Vec<f32> = hz_points
        .iter()
        .map(|&hz| (hz / sample_rate as f32) * (n as f32))
        .collect();

    let mut mel_energies = vec![0.0f32; n_mels];
    for i in 0..n_mels {
        let left = bin_points[i] as usize;
        let center = bin_points[i + 1] as usize;
        let right = bin_points[i + 2] as usize;
        let left = left.max(1);
        let right = right.min(spectrum_len - 1);
        let center = center.max(left).min(right);

        let mut energy = 0.0f32;
        for k in left..=right {
            let weight = if k <= center && center > left {
                (k - left) as f32 / (center - left) as f32
            } else if k > center && right > center {
                (right - k) as f32 / (right - center) as f32
            } else {
                1.0
            };
            energy += power[k] * weight;
        }
        mel_energies[i] = energy.max(1e-10).ln(); // log 压缩
    }

    // L2 归一化
    let norm: f32 = mel_energies.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm < 1e-10 {
        return vec![0.0; n_mels];
    }
    mel_energies.iter().map(|&x| x / norm).collect()
}

fn hz_to_mel(hz: f32) -> f32 {
    2595.0 * (1.0 + hz / 700.0).ln() / std::f32::consts::LN_10
}

fn mel_to_hz(mel: f32) -> f32 {
    700.0 * (10.0f32.powf(mel / 2595.0) - 1.0)
}

/// 余弦相似度
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a < 1e-10 || norm_b < 1e-10 {
        return 0.0;
    }
    dot / (norm_a * norm_b)
}

/// 计算帧与参考特征的最大相似度
fn compute_max_similarity(frame_features: &[f32], reference: &[Vec<f32>]) -> f32 {
    reference
        .iter()
        .map(|ref_feat| cosine_similarity(frame_features, ref_feat))
        .fold(0.0f32, f32::max)
}

// ============================================================
// 汉宁窗
// ============================================================

fn hanning_window(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| 0.5 * (1.0 - (2.0 * std::f32::consts::PI * i as f32 / n as f32).cos()))
        .collect()
}

// ============================================================
// 音频处理器
// ============================================================

struct AudioProcessor {
    mode: ProcessingMode,
    targets: Vec<TargetVoice>,
    // 帧参数
    frame_size: usize,
    hop_size: usize,
    sample_rate: u32,
    n_mels: usize,
    // 缓冲区
    input_buffer: Vec<f32>,
    overlap_buffer: Vec<f32>,
    // 增益平滑
    current_gain: f32,
    // 消除保持
    suppress_hold_frames: usize,
    suppress_hold_counter: usize,
    // 时间平滑
    similarity_history: Vec<Vec<f32>>,
    // FFT planner
    fft_planner: FftPlanner<f32>,
    // 窗函数
    window: Vec<f32>,
    // 统计
    frames_processed: u64,
    // 调试
    debug_counter: u64,
}

impl AudioProcessor {
    fn new(mode: ProcessingMode, targets: Vec<TargetVoice>, actual_sample_rate: u32) -> Self {
        // 根据目标时间帧长计算实际帧长（不强制2的幂）
        let frame_size = (actual_sample_rate as f64 * mode.frame_duration_ms() as f64 / 1000.0).round() as usize;
        let frame_size = frame_size.max(64); // 最小64样本
        let hop_size = (frame_size as f32 * (1.0 - mode.overlap_ratio())) as usize;
        let hop_size = hop_size.max(1);

        let window = if mode.uses_window() {
            hanning_window(frame_size)
        } else {
            vec![1.0; frame_size]
        };

        let overlap_len = frame_size - hop_size;
        let overlap_buffer = if mode.uses_overlap() {
            vec![0.0; overlap_len]
        } else {
            vec![]
        };

        let temporal_frames = mode.temporal_smooth_frames();
        let num_targets = targets.len();
        let similarity_history = vec![vec![0.0; num_targets]; temporal_frames];

        AudioProcessor {
            mode,
            targets,
            frame_size,
            hop_size,
            sample_rate: actual_sample_rate,
            n_mels: 32,
            input_buffer: Vec::with_capacity(frame_size * 2),
            overlap_buffer,
            current_gain: 1.0,
            suppress_hold_frames: 3,
            suppress_hold_counter: 0,
            similarity_history,
            fft_planner: FftPlanner::new(),
            window,
            frames_processed: 0,
            debug_counter: 0,
        }
    }

    /// 计算帧的RMS能量
    fn frame_rms(frame: &[f32]) -> f32 {
        let sum: f32 = frame.iter().map(|&x| x * x).sum();
        (sum / frame.len() as f32).sqrt()
    }

    /// 提取当前帧的特征
    fn extract_features(&mut self, frame: &[f32]) -> Vec<f32> {
        if self.mode.uses_fft() {
            let windowed: Vec<f32> = frame
                .iter()
                .zip(self.window.iter())
                .map(|(&s, &w)| s * w)
                .collect();
            extract_mel_spectrum(&windowed, &mut self.fft_planner, self.sample_rate, self.n_mels)
        } else {
            extract_subband_energy(frame)
        }
    }

    /// 计算每个目标的平滑相似度
    fn compute_smoothed_similarities(&mut self, frame_features: &[f32]) -> Vec<f32> {
        let n_targets = self.targets.len();
        let mut current_sims = Vec::with_capacity(n_targets);

        for target in &self.targets {
            let sim = compute_max_similarity(frame_features, &target.reference_features);
            current_sims.push(sim);
        }

        // 更新历史
        self.similarity_history.pop();
        self.similarity_history.insert(0, current_sims.clone());

        // 时间平滑：取最近几帧的最大值
        let temporal_frames = self.mode.temporal_smooth_frames();
        let mut smoothed = vec![0.0f32; n_targets];
        for t in 0..n_targets {
            let mut max_sim = 0.0f32;
            let count = self.similarity_history.len().min(temporal_frames);
            for h in 0..count {
                max_sim = max_sim.max(self.similarity_history[h][t]);
            }
            smoothed[t] = max_sim;
        }

        smoothed
    }

    /// 决定增益
    fn decide_gain(&mut self, similarities: &[f32]) -> f32 {
        let threshold = self.mode.similarity_threshold();

        // 优先检查消除目标
        for (i, target) in self.targets.iter().enumerate() {
            if target.action == TargetAction::Suppress && similarities[i] >= threshold {
                self.suppress_hold_counter = self.suppress_hold_frames;
                return self.mode.suppress_gain();
            }
        }

        // 消除保持：即使当前帧未匹配，也保持消除若干帧
        if self.suppress_hold_counter > 0 {
            self.suppress_hold_counter -= 1;
            return self.mode.suppress_gain();
        }

        // 再检查增强目标
        for (i, target) in self.targets.iter().enumerate() {
            if target.action == TargetAction::Enhance && similarities[i] >= threshold {
                return self.mode.enhance_gain();
            }
        }

        1.0
    }

    /// 处理一帧音频
    fn process_frame(&mut self, frame: &mut [f32]) {
        debug_assert_eq!(frame.len(), self.frame_size);

        // 噪声门限：静音帧直接跳过处理
        let rms = Self::frame_rms(frame);
        if rms < 0.001 {
            // 静音帧：重置增益到1.0，输出静音
            self.current_gain = 1.0;
            self.suppress_hold_counter = 0;
            self.frames_processed += 1;
            return;
        }

        // 提取特征
        let features = self.extract_features(frame);

        // 计算平滑相似度
        let similarities = self.compute_smoothed_similarities(&features);

        // 决定目标增益
        let target_gain = self.decide_gain(&similarities);

        // 增益平滑（一阶低通滤波）
        let alpha = self.mode.smoothing_coeff();
        self.current_gain = alpha * self.current_gain + (1.0 - alpha) * target_gain;

        // 每100帧打印调试信息
        self.debug_counter += 1;
        if self.debug_counter % 100 == 0 {
            eprint!("\r[调试] 相似度: ");
            for (i, t) in self.targets.iter().enumerate() {
                eprint!("{}={:.3} ", t.name, similarities[i]);
            }
            eprint!("| 增益: {:.3} ", self.current_gain);
        }

        // 应用增益
        for sample in frame.iter_mut() {
            *sample *= self.current_gain;
        }

        self.frames_processed += 1;
    }

    /// 处理输入样本流，返回输出样本
    fn process_samples(&mut self, input: &[f32]) -> Vec<f32> {
        let frame_size = self.frame_size;
        let hop_size = self.hop_size;

        self.input_buffer.extend_from_slice(input);

        let mut output = Vec::new();

        while self.input_buffer.len() >= frame_size {
            let mut frame: Vec<f32> = self.input_buffer[..frame_size].to_vec();
            self.input_buffer.drain(..hop_size);

            self.process_frame(&mut frame);

            if self.mode.uses_overlap() {
                let overlap_len = self.overlap_buffer.len();

                for i in 0..overlap_len {
                    frame[i] += self.overlap_buffer[i];
                }

                self.overlap_buffer
                    .copy_from_slice(&frame[hop_size..frame_size]);

                output.extend_from_slice(&frame[..hop_size]);
            } else {
                output.extend_from_slice(&frame);
            }
        }

        output
    }
}

// ============================================================
// 参考录音
// ============================================================

fn record_reference(
    device: &cpal::Device,
    config: &StreamConfig,
    duration_secs: u64,
) -> Result<Vec<f32>, String> {
    let sample_rate = config.sample_rate.0;
    let channels = config.channels as usize;
    let total_samples = sample_rate as usize * channels * duration_secs as usize;

    let recorded = Arc::new(Mutex::new(Vec::with_capacity(total_samples)));
    let recorded_clone = recorded.clone();

    let stream = device
        .build_input_stream(
            config,
            move |data: &[f32], _: &cpal::InputCallbackInfo| {
                let mut buf = recorded_clone.lock().unwrap();
                buf.extend_from_slice(data);
            },
            |err| eprintln!("录音错误: {}", err),
            None,
        )
        .map_err(|e| format!("无法创建录音流: {}", e))?;

    stream.play().map_err(|e| format!("无法开始录音: {}", e))?;

    println!("  录音中... 请说话（{}秒）", duration_secs);
    std::thread::sleep(std::time::Duration::from_secs(duration_secs));

    drop(stream);

    let samples = Arc::try_unwrap(recorded).unwrap().into_inner().unwrap();

    let mono = if channels > 1 {
        samples.iter().step_by(channels).copied().collect()
    } else {
        samples
    };

    Ok(mono)
}

/// 从录音中提取参考特征
fn build_reference_features(
    audio: &[f32],
    mode: ProcessingMode,
    actual_sample_rate: u32,
) -> Vec<Vec<f32>> {
    let frame_size = (actual_sample_rate as f64 * mode.frame_duration_ms() as f64 / 1000.0).round() as usize;
    let frame_size = frame_size.max(64);
    let hop_size = frame_size;
    let n_mels = 32;
    let mut planner = FftPlanner::new();
    let window = if mode.uses_window() {
        hanning_window(frame_size)
    } else {
        vec![1.0; frame_size]
    };

    let mut features = Vec::new();
    let mut pos = 0;

    while pos + frame_size <= audio.len() {
        let frame = &audio[pos..pos + frame_size];

        // 跳过静音帧
        let rms = (frame.iter().map(|&x| x * x).sum::<f32>() / frame.len() as f32).sqrt();
        if rms < 0.001 {
            pos += hop_size;
            continue;
        }

        let feat = if mode.uses_fft() {
            let windowed: Vec<f32> = frame
                .iter()
                .zip(window.iter())
                .map(|(&s, &w)| s * w)
                .collect();
            extract_mel_spectrum(&windowed, &mut planner, actual_sample_rate, n_mels)
        } else {
            extract_subband_energy(frame)
        };

        features.push(feat);
        pos += hop_size;
    }

    println!(
        "  参考特征: {} 帧, 维度: {} (梅尔频段)",
        features.len(),
        features.first().map(|f| f.len()).unwrap_or(0)
    );

    // 限制参考特征数量
    let max_features = 100;
    if features.len() > max_features {
        let step = features.len() as f32 / max_features as f32;
        let sampled: Vec<Vec<f32>> = (0..max_features)
            .map(|i| features[(i as f32 * step) as usize].clone())
            .collect();
        sampled
    } else {
        features
    }
}

// ============================================================
// 用户交互
// ============================================================

fn read_line(prompt: &str) -> String {
    print!("{}", prompt);
    io::stdout().flush().unwrap();
    let mut input = String::new();
    io::stdin().read_line(&mut input).unwrap();
    input.trim().to_string()
}

fn select_processing_mode() -> ProcessingMode {
    loop {
        let choice = read_line(
            "\n选择全局处理模式:\n  1) 性能模式 (低延迟, 低CPU, 单目标)\n  2) 平衡模式 (中等质量, 多目标)\n  3) 深度模式 (高质量, 多目标)\n请输入 (1/2/3): ",
        );
        match choice.as_str() {
            "1" => {
                println!("已选择: 性能模式");
                return ProcessingMode::Performance;
            }
            "2" => {
                println!("已选择: 平衡模式");
                return ProcessingMode::Balanced;
            }
            "3" => {
                println!("已选择: 深度模式");
                return ProcessingMode::Deep;
            }
            _ => println!("无效选择，请重新输入"),
        }
    }
}

fn select_target_action(name: &str) -> TargetAction {
    loop {
        let choice = read_line(&format!(
            "  对 \"{}\" 的处理模式 - 消除(1) / 增强(2): ",
            name
        ));
        match choice.as_str() {
            "1" => {
                println!("  → 将消除 \"{}\" 的人声", name);
                return TargetAction::Suppress;
            }
            "2" => {
                println!("  → 将增强 \"{}\" 的人声", name);
                return TargetAction::Enhance;
            }
            _ => println!("  无效选择，请输入 1 或 2"),
        }
    }
}

// ============================================================
// 主函数
// ============================================================

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== VocalSieve - 实时人声筛选器 ===\n");

    let mode = select_processing_mode();

    let mut targets: Vec<TargetVoice> = Vec::new();

    let host = cpal::default_host();
    let input_device = host
        .default_input_device()
        .ok_or("未找到输入音频设备")?;
    let output_device = host
        .default_output_device()
        .ok_or("未找到输出音频设备")?;

    println!("\n输入设备: {}", input_device.name().unwrap_or_default());
    println!("输出设备: {}", output_device.name().unwrap_or_default());

    let default_input_config = input_device.default_input_config()?;
    let default_output_config = output_device.default_output_config()?;

    let input_sample_rate = default_input_config.sample_rate().0;
    let output_sample_rate = default_output_config.sample_rate().0;

    println!("输入采样率: {} Hz", input_sample_rate);
    println!("输出采样率: {} Hz", output_sample_rate);

    let input_config: StreamConfig = default_input_config.into();
    let output_config: StreamConfig = default_output_config.into();

    // 添加目标循环
    loop {
        let max = mode.max_targets();
        if targets.len() >= max {
            println!("\n已达到该模式下最大目标数量 ({})", max);
            break;
        }

        if !targets.is_empty() {
            println!("\n已添加 {} 个目标", targets.len());
        }

        let add_more = read_line("是否添加目标人声? (y/n): ");
        if add_more.to_lowercase() != "y" {
            break;
        }

        let name = read_line("  输入目标名称: ");
        if name.is_empty() {
            println!("  名称不能为空");
            continue;
        }

        let action = select_target_action(&name);

        println!("  准备录制参考声音...");
        std::thread::sleep(std::time::Duration::from_millis(500));

        let ref_audio = record_reference(&input_device, &input_config, 5)?;

        if ref_audio.len() < 128 {
            println!("  录音太短，请重试");
            continue;
        }

        let features = build_reference_features(&ref_audio, mode, input_sample_rate);
        println!(
            "  已提取 {} 帧参考特征 (维度: {})",
            features.len(),
            features.first().map(|f| f.len()).unwrap_or(0)
        );

        targets.push(TargetVoice {
            name,
            action,
            reference_features: features,
        });
    }

    if targets.is_empty() {
        println!("\n未添加任何目标，退出。");
        return Ok(());
    }

    let processor = Arc::new(Mutex::new(AudioProcessor::new(mode, targets, output_sample_rate)));

    let proc_frame_size = processor.lock().unwrap().frame_size;
    let proc_hop_size = processor.lock().unwrap().hop_size;
    println!("\n=== 配置摘要 ===");
    println!("模式: {:?}", mode);
    println!("输入采样率: {} Hz", input_sample_rate);
    println!("输出采样率: {} Hz", output_sample_rate);
    println!("帧长: {} 样本 ({:.1} ms)", proc_frame_size, proc_frame_size as f32 / output_sample_rate as f32 * 1000.0);
    println!("帧移: {} 样本 ({:.1} ms)", proc_hop_size, proc_hop_size as f32 / output_sample_rate as f32 * 1000.0);
    println!("目标数量: {}", processor.lock().unwrap().targets.len());
    for t in &processor.lock().unwrap().targets {
        println!(
            "  - {} ({})",
            t.name,
            match t.action {
                TargetAction::Suppress => "消除",
                TargetAction::Enhance => "增强",
            }
        );
    }
    println!("================\n");

    let (tx, rx) = std::sync::mpsc::sync_channel::<Vec<f32>>(8);

    let processor_in = processor.clone();
    let tx_in = tx.clone();
    let input_stream = build_input_stream(&input_device, &input_config, processor_in, tx_in, output_sample_rate)?;

    let output_stream = build_output_stream(&output_device, &output_config, rx)?;

    input_stream.play()?;
    output_stream.play()?;

    println!("实时处理已启动，按 Ctrl+C 退出...");
    println!("提示: 建议使用耳机避免扬声器反馈");

    let running = Arc::new(Mutex::new(true));
    let r = running.clone();
    ctrlc_handler(r);

    while *running.lock().unwrap() {
        std::thread::sleep(std::time::Duration::from_millis(100));
    }

    println!("\n正在停止...");

    drop(input_stream);
    drop(output_stream);

    let proc = processor.lock().unwrap();
    println!("已处理 {} 帧", proc.frames_processed);
    println!("VocalSieve 已退出。");

    Ok(())
}

fn build_input_stream(
    device: &cpal::Device,
    config: &StreamConfig,
    processor: Arc<Mutex<AudioProcessor>>,
    tx: std::sync::mpsc::SyncSender<Vec<f32>>,
    output_sample_rate: u32,
) -> Result<Stream, Box<dyn std::error::Error>> {
    let channels = config.channels as usize;
    let input_sample_rate = config.sample_rate.0;
    let need_resample = input_sample_rate != output_sample_rate;
    let ratio = output_sample_rate as f64 / input_sample_rate as f64;

    let stream = device.build_input_stream(
        config,
        move |data: &[f32], _: &cpal::InputCallbackInfo| {
            let mono: Vec<f32> = if channels > 1 {
                data.chunks(channels).map(|ch| ch[0]).collect()
            } else {
                data.to_vec()
            };

            let samples = if need_resample {
                let input_len = mono.len();
                let output_len = (input_len as f64 * ratio) as usize;
                let mut resampled = Vec::with_capacity(output_len);
                for i in 0..output_len {
                    let src_pos = i as f64 / ratio;
                    let idx = src_pos as usize;
                    let frac = src_pos - idx as f64;
                    let s0 = mono.get(idx).copied().unwrap_or(0.0);
                    let s1 = mono.get(idx + 1).copied().unwrap_or(0.0);
                    resampled.push(s0 + (s1 - s0) * frac as f32);
                }
                resampled
            } else {
                mono
            };

            let mut proc = processor.lock().unwrap();
            let output = proc.process_samples(&samples);

            if !output.is_empty() {
                // sync_channel 满时丢弃旧数据，避免延迟累积
                let _ = tx.try_send(output);
            }
        },
        |err| eprintln!("输入流错误: {}", err),
        None,
    )?;

    Ok(stream)
}

fn build_output_stream(
    device: &cpal::Device,
    config: &StreamConfig,
    rx: std::sync::mpsc::Receiver<Vec<f32>>,
) -> Result<Stream, Box<dyn std::error::Error>> {
    let channels = config.channels as usize;
    let sample_rate = config.sample_rate.0;
    // 限制输出缓冲区大小，防止延迟累积（最多100ms的数据）
    let max_output_samples = sample_rate as usize * channels / 10;
    let mut output_buffer: VecDeque<f32> = VecDeque::with_capacity(max_output_samples);

    let stream = device.build_output_stream(
        config,
        move |data: &mut [f32], _: &cpal::OutputCallbackInfo| {
            // 从通道获取处理后的数据
            while let Ok(chunk) = rx.try_recv() {
                output_buffer.extend(chunk);
            }

            // 如果缓冲区过大，丢弃旧数据
            while output_buffer.len() > max_output_samples {
                output_buffer.pop_front();
            }

            // 填充输出缓冲区
            for frame in data.chunks_mut(channels) {
                let sample = output_buffer.pop_front().unwrap_or(0.0);
                for ch in frame.iter_mut() {
                    *ch = sample;
                }
            }
        },
        |err| eprintln!("输出流错误: {}", err),
        None,
    )?;

    Ok(stream)
}

fn ctrlc_handler(running: Arc<Mutex<bool>>) {
    ctrlc::set_handler(move || {
        *running.lock().unwrap() = false;
    }).unwrap();
}
