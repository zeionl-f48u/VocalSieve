// ============================================================
// VocalSieve - 实时人声筛选器 · 音频处理模块
// ============================================================
// 本模块实现核心音频处理逻辑，包括：
//   - 三种处理模式（性能/平衡/深度）及其参数
//   - 目标人声的数据结构与持久化
//   - 特征提取（子带能量比、梅尔频谱）
//   - 相似度计算与时间平滑
//   - 实时音频处理器（分帧、增益决策、重叠相加）
//   - 参考录音与特征构建
//   - 音频流管理（输入采集、输出播放、跨设备重采样）
//   - 虚拟音频线缆检测
//   - 设备枚举与查找
// ============================================================

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{Stream, StreamConfig};
use rustfft::{num_complex::Complex, FftPlanner};
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

// ============================================================
// 处理模式
// ============================================================

/// 处理模式枚举，控制音频处理的精度与性能权衡
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub enum ProcessingMode {
    /// 性能模式：使用子带能量比（4维），无FFT，最低延迟
    Performance,
    /// 平衡模式：使用梅尔频谱（32维），50%重叠，中等延迟
    Balanced,
    /// 深度模式：使用梅尔频谱（32维）+汉宁窗+87.5%重叠，最高精度
    Deep,
}

impl ProcessingMode {
    /// 返回模式的中文标签
    pub fn label(&self) -> &str {
        match self {
            Self::Performance => "性能模式",
            Self::Balanced => "平衡模式",
            Self::Deep => "深度模式",
        }
    }

    /// 返回所有可选模式的切片
    pub fn all() -> &'static [ProcessingMode] {
        &[Self::Performance, Self::Balanced, Self::Deep]
    }

    /// 每帧音频的时长（毫秒），决定帧大小
    pub fn frame_duration_ms(&self) -> f32 {
        match self {
            Self::Performance => 16.0,  // ~16ms，低延迟
            Self::Balanced => 32.0,     // ~32ms，平衡
            Self::Deep => 46.0,         // ~46ms，高精度
        }
    }

    /// 帧间重叠比例，重叠越多过渡越平滑但计算量越大
    pub fn overlap_ratio(&self) -> f32 {
        match self {
            Self::Performance => 0.0,   // 无重叠
            Self::Balanced => 0.5,      // 50%重叠
            Self::Deep => 0.875,        // 87.5%重叠（8倍冗余）
        }
    }

    /// 余弦相似度判定阈值，超过此值认为匹配到目标
    pub fn similarity_threshold(&self) -> f32 {
        match self {
            Self::Performance => 0.70,  // 高阈值，减少误判
            Self::Balanced => 0.65,     // 中等阈值
            Self::Deep => 0.55,         // 低阈值，更灵敏
        }
    }

    /// 消除模式下的增益值，0.0 = 完全静音
    pub fn suppress_gain(&self) -> f32 {
        0.0
    }

    /// 增强模式下的增益倍数
    pub fn enhance_gain(&self) -> f32 {
        match self {
            Self::Performance => 2.5,
            Self::Balanced => 2.0,
            Self::Deep => 3.0,
        }
    }

    /// 增益平滑系数（一阶低通滤波），越大则增益变化越缓慢
    pub fn smoothing_coeff(&self) -> f32 {
        match self {
            Self::Performance => 0.70,
            Self::Balanced => 0.60,
            Self::Deep => 0.80,
        }
    }

    /// 时间平滑帧数：取最近N帧中相似度的最大值，防止瞬态抖动
    pub fn temporal_smooth_frames(&self) -> usize {
        match self {
            Self::Performance => 1,  // 无时间平滑
            Self::Balanced => 3,     // 3帧平滑
            Self::Deep => 5,         // 5帧平滑
        }
    }

    /// 该模式支持的最大消除目标数
    pub fn max_targets(&self) -> usize {
        match self {
            Self::Performance => 1,       // 性能模式仅支持1个消除目标
            Self::Balanced => usize::MAX, // 无限制
            Self::Deep => usize::MAX,     // 无限制
        }
    }

    /// 是否使用FFT进行频谱分析（性能模式使用子带能量比代替）
    pub fn uses_fft(&self) -> bool {
        !matches!(self, Self::Performance)
    }

    /// 是否使用汉宁窗减少频谱泄漏
    pub fn uses_window(&self) -> bool {
        matches!(self, Self::Deep)
    }

    /// 是否使用重叠相加法实现帧间平滑过渡
    pub fn uses_overlap(&self) -> bool {
        !matches!(self, Self::Performance)
    }
}

// ============================================================
// 目标人声
// ============================================================

/// 对目标人声执行的动作类型
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub enum TargetAction {
    /// 消除（抑制）：将匹配到的目标人声静音
    Suppress,
    /// 增强（放大）：将匹配到的目标人声增益
    Enhance,
}

impl TargetAction {
    /// 返回动作的中文标签
    pub fn label(&self) -> &str {
        match self {
            Self::Suppress => "消除",
            Self::Enhance => "增强",
        }
    }
}

/// 目标人声数据结构
/// 保存目标名称、动作类型和参考特征向量集合
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TargetVoice {
    /// 目标人声的名称标识
    pub name: String,
    /// 对该目标执行的动作
    pub action: TargetAction,
    /// 参考特征向量集合，每帧提取一个特征向量
    #[serde(with = "vec_of_vec_f32")]
    pub reference_features: Vec<Vec<f32>>,
}

/// serde 辅助模块：为 Vec<Vec<f32>> 提供序列化/反序列化支持
mod vec_of_vec_f32 {
    use serde::{Deserialize, Deserializer, Serializer};

    pub fn serialize<S: Serializer>(data: &Vec<Vec<f32>>, s: S) -> Result<S::Ok, S::Error> {
        s.collect_seq(data.iter().map(|v| v.iter().collect::<Vec<_>>()))
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<Vec<Vec<f32>>, D::Error> {
        let raw: Vec<Vec<f32>> = Vec::deserialize(d)?;
        Ok(raw)
    }
}

// ============================================================
// 特征提取
// ============================================================

/// 子带能量比特征提取（性能模式专用）
/// 将一帧音频等分为4个子带，计算各子带能量占总能量的比例
/// 返回4维特征向量，值域 [0, 1]
fn extract_subband_energy(frame: &[f32]) -> Vec<f32> {
    let n = frame.len();
    let band1_end = n / 4;
    let band2_end = n / 2;
    let band3_end = 3 * n / 4;

    // 计算各子带能量（平方和）
    let e1: f32 = frame[..band1_end].iter().map(|&x| x * x).sum();
    let e2: f32 = frame[band1_end..band2_end].iter().map(|&x| x * x).sum();
    let e3: f32 = frame[band2_end..band3_end].iter().map(|&x| x * x).sum();
    let e4: f32 = frame[band3_end..].iter().map(|&x| x * x).sum();

    let total = e1 + e2 + e3 + e4;
    if total < 1e-10 {
        // 静音帧返回均匀分布
        return vec![0.25, 0.25, 0.25, 0.25];
    }
    // 返回各子带能量占比
    vec![e1 / total, e2 / total, e3 / total, e4 / total]
}

/// 梅尔频谱特征提取（平衡/深度模式使用）
/// 对一帧音频执行FFT，将功率谱映射到梅尔滤波器组，再进行Log压缩和L2归一化
/// 返回 n_mels 维特征向量
fn extract_mel_spectrum(
    frame: &[f32],
    planner: &mut FftPlanner<f32>,
    sample_rate: u32,
    n_mels: usize,
) -> Vec<f32> {
    let n = frame.len();
    // 执行FFT变换
    let fft = planner.plan_fft_forward(n);
    let mut buffer: Vec<Complex<f32>> = frame.iter().map(|&x| Complex::new(x, 0.0)).collect();
    fft.process(&mut buffer);

    // 取前半部分（单边频谱），计算功率谱
    let spectrum_len = n / 2 + 1;
    let power: Vec<f32> = buffer[..spectrum_len]
        .iter()
        .map(|c| c.norm_sqr() / (n as f32))
        .collect();

    // 构建梅尔滤波器组的频率点
    let mel_low = hz_to_mel(80.0);   // 最低频率80Hz，去除低频噪声
    let mel_high = hz_to_mel(sample_rate as f32 / 2.0); // 最高到奈奎斯特频率
    // 在梅尔刻度上均匀分布 n_mels+2 个点（含两端边界）
    let mel_points: Vec<f32> = (0..=n_mels + 1)
        .map(|i| mel_low + (mel_high - mel_low) * i as f32 / (n_mels + 1) as f32)
        .collect();
    // 转回Hz频率
    let hz_points: Vec<f32> = mel_points.iter().map(|&m| mel_to_hz(m)).collect();
    // 转换为FFT bin索引
    let bin_points: Vec<f32> = hz_points
        .iter()
        .map(|&hz| (hz / sample_rate as f32) * (n as f32))
        .collect();

    // 应用三角滤波器组，计算每个梅尔频带的能量
    let mut mel_energies = vec![0.0f32; n_mels];
    for i in 0..n_mels {
        let left = bin_points[i] as usize;
        let center = bin_points[i + 1] as usize;
        let right = bin_points[i + 2] as usize;
        // 边界保护
        let left = left.max(1);
        let right = right.min(spectrum_len - 1);
        let center = center.max(left).min(right);

        // 三角滤波：从left到center线性上升，从center到right线性下降
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
        // Log压缩：将能量转换为对数刻度，更符合人耳感知
        mel_energies[i] = energy.max(1e-10).ln();
    }

    // L2归一化：消除音量差异，使特征只反映频谱形状
    let norm: f32 = mel_energies.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm < 1e-10 {
        return vec![0.0; n_mels];
    }
    mel_energies.iter().map(|&x| x / norm).collect()
}

/// Hz频率转梅尔刻度
/// 梅尔刻度模拟人耳对频率的非线性感知
fn hz_to_mel(hz: f32) -> f32 {
    2595.0 * (1.0 + hz / 700.0).ln() / std::f32::consts::LN_10
}

/// 梅尔刻度转Hz频率
fn mel_to_hz(mel: f32) -> f32 {
    700.0 * (10.0f32.powf(mel / 2595.0) - 1.0)
}

/// 计算两个特征向量的余弦相似度
/// 返回值域 [-1, 1]，1表示完全相同，0表示正交（无关）
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

/// 计算当前帧特征与参考特征集合的最大相似度
/// 遍历所有参考帧，返回最高相似度值
fn compute_max_similarity(frame_features: &[f32], reference: &[Vec<f32>]) -> f32 {
    reference
        .iter()
        .map(|ref_feat| cosine_similarity(frame_features, ref_feat))
        .fold(0.0f32, f32::max)
}

/// 生成汉宁窗函数
/// 用于减少分帧造成的频谱泄漏
fn hanning_window(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| 0.5 * (1.0 - (2.0 * std::f32::consts::PI * i as f32 / n as f32).cos()))
        .collect()
}

// ============================================================
// 音频处理器
// ============================================================

/// 实时音频处理器
/// 接收原始音频样本，执行分帧→特征提取→相似度计算→增益决策→帧处理→重叠相加的完整流水线
pub struct AudioProcessor {
    /// 当前处理模式
    mode: ProcessingMode,
    /// 目标人声列表
    targets: Vec<TargetVoice>,
    /// 帧大小（样本数），由采样率和帧时长决定
    frame_size: usize,
    /// 帧移大小（样本数），frame_size * (1 - overlap_ratio)
    hop_size: usize,
    /// 实际使用的采样率
    sample_rate: u32,
    /// 梅尔频谱的频带数
    n_mels: usize,
    /// 输入样本缓冲区，累积到frame_size后处理
    input_buffer: Vec<f32>,
    /// 重叠相加缓冲区，保存上一帧的尾部用于与下一帧叠加
    overlap_buffer: Vec<f32>,
    /// 当前增益值，通过一阶低通滤波平滑过渡
    current_gain: f32,
    /// 消除保持帧数：触发消除后即使相似度下降也保持消除的帧数
    suppress_hold_frames: usize,
    /// 消除保持计数器
    suppress_hold_counter: usize,
    /// 相似度历史记录，用于时间平滑（取多帧最大值）
    similarity_history: Vec<Vec<f32>>,
    /// FFT计划器，缓存FFT算法以避免重复计算
    fft_planner: FftPlanner<f32>,
    /// 窗函数系数
    window: Vec<f32>,
    /// 已处理的帧计数
    frames_processed: u64,
    /// 最近一次计算的各目标相似度（供UI读取）
    pub last_similarities: Vec<f32>,
    /// 最近一次的增益值（供UI读取）
    pub last_gain: f32,
}

// 安全声明：AudioProcessor的FftPlanner内部使用RefCell，
// 但我们保证同一时间只在一个线程访问，因此可以安全跨线程传递
unsafe impl Send for AudioProcessor {}

impl AudioProcessor {
    /// 创建新的音频处理器
    /// actual_sample_rate: 实际输出设备的采样率，用于计算帧大小
    pub fn new(mode: ProcessingMode, targets: Vec<TargetVoice>, actual_sample_rate: u32) -> Self {
        // 根据采样率和帧时长计算帧大小
        let frame_size = (actual_sample_rate as f64 * mode.frame_duration_ms() as f64 / 1000.0).round() as usize;
        let frame_size = frame_size.max(64); // 最小64样本
        // 帧移 = 帧大小 × (1 - 重叠比例)
        let hop_size = (frame_size as f32 * (1.0 - mode.overlap_ratio())) as usize;
        let hop_size = hop_size.max(1);

        // 根据模式选择窗函数
        let window = if mode.uses_window() {
            hanning_window(frame_size)
        } else {
            vec![1.0; frame_size] // 无窗等效于矩形窗
        };

        // 重叠缓冲区大小 = 帧大小 - 帧移
        let overlap_len = frame_size - hop_size;
        let overlap_buffer = if mode.uses_overlap() {
            vec![0.0; overlap_len]
        } else {
            vec![]
        };

        // 初始化相似度历史环形缓冲区
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
            suppress_hold_frames: 3, // 触发消除后保持3帧
            suppress_hold_counter: 0,
            similarity_history,
            fft_planner: FftPlanner::new(),
            window,
            frames_processed: 0,
            last_similarities: vec![0.0; num_targets],
            last_gain: 1.0,
        }
    }

    /// 计算一帧音频的RMS（均方根）能量
    fn frame_rms(frame: &[f32]) -> f32 {
        let sum: f32 = frame.iter().map(|&x| x * x).sum();
        (sum / frame.len() as f32).sqrt()
    }

    /// 根据当前模式提取帧特征
    /// 性能模式：子带能量比（4维）
    /// 平衡/深度模式：梅尔频谱（32维）
    fn extract_features(&mut self, frame: &[f32]) -> Vec<f32> {
        if self.mode.uses_fft() {
            // 加窗后提取梅尔频谱
            let windowed: Vec<f32> = frame
                .iter()
                .zip(self.window.iter())
                .map(|(&s, &w)| s * w)
                .collect();
            extract_mel_spectrum(&windowed, &mut self.fft_planner, self.sample_rate, self.n_mels)
        } else {
            // 性能模式直接提取子带能量比
            extract_subband_energy(frame)
        }
    }

    /// 计算时间平滑后的相似度
    /// 对每个目标，取最近N帧中相似度的最大值，防止瞬态抖动
    fn compute_smoothed_similarities(&mut self, frame_features: &[f32]) -> Vec<f32> {
        let n_targets = self.targets.len();
        // 计算当前帧与每个目标的相似度
        let mut current_sims = Vec::with_capacity(n_targets);
        for target in &self.targets {
            let sim = compute_max_similarity(frame_features, &target.reference_features);
            current_sims.push(sim);
        }

        // 更新历史：移除最旧的记录，插入最新的
        self.similarity_history.pop();
        self.similarity_history.insert(0, current_sims.clone());

        // 对每个目标取时间窗口内的最大相似度
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

    /// 根据相似度决定增益值
    /// 优先级：消除 > 消除保持 > 增强 > 默认（1.0）
    fn decide_gain(&mut self, similarities: &[f32]) -> f32 {
        let threshold = self.mode.similarity_threshold();

        // 第一优先级：检测消除目标
        for (i, target) in self.targets.iter().enumerate() {
            if target.action == TargetAction::Suppress && similarities[i] >= threshold {
                self.suppress_hold_counter = self.suppress_hold_frames;
                return self.mode.suppress_gain();
            }
        }

        // 第二优先级：消除保持期（防止消除后立即恢复导致的"回音"感）
        if self.suppress_hold_counter > 0 {
            self.suppress_hold_counter -= 1;
            return self.mode.suppress_gain();
        }

        // 第三优先级：检测增强目标
        for (i, target) in self.targets.iter().enumerate() {
            if target.action == TargetAction::Enhance && similarities[i] >= threshold {
                return self.mode.enhance_gain();
            }
        }

        // 默认：不修改增益
        1.0
    }

    /// 处理单帧音频
    /// 流水线：噪声门 → 特征提取 → 相似度计算 → 增益决策 → 增益平滑 → 应用增益
    fn process_frame(&mut self, frame: &mut [f32]) {
        debug_assert_eq!(frame.len(), self.frame_size);

        // 噪声门：RMS过低的静音帧跳过处理
        let rms = Self::frame_rms(frame);
        if rms < 0.001 {
            self.current_gain = 1.0;
            self.suppress_hold_counter = 0;
            self.last_gain = 1.0;
            self.frames_processed += 1;
            return;
        }

        // 特征提取
        let features = self.extract_features(frame);
        // 相似度计算（含时间平滑）
        let similarities = self.compute_smoothed_similarities(&features);
        // 增益决策
        let target_gain = self.decide_gain(&similarities);

        // 一阶低通滤波平滑增益，避免增益突变产生爆音
        let alpha = self.mode.smoothing_coeff();
        self.current_gain = alpha * self.current_gain + (1.0 - alpha) * target_gain;

        // 保存实时状态供UI读取
        self.last_similarities = similarities;
        self.last_gain = self.current_gain;

        // 应用增益到每个样本
        for sample in frame.iter_mut() {
            *sample *= self.current_gain;
        }

        self.frames_processed += 1;
    }

    /// 处理一批输入样本，返回处理后的输出样本
    /// 实现分帧、帧处理和重叠相加的完整流程
    pub fn process_samples(&mut self, input: &[f32]) -> Vec<f32> {
        let frame_size = self.frame_size;
        let hop_size = self.hop_size;

        // 将新样本追加到输入缓冲区
        self.input_buffer.extend_from_slice(input);

        let mut output = Vec::new();

        // 当缓冲区中样本数足够一帧时，循环处理
        while self.input_buffer.len() >= frame_size {
            // 取出一帧
            let mut frame: Vec<f32> = self.input_buffer[..frame_size].to_vec();
            // 按帧移大小前移缓冲区
            self.input_buffer.drain(..hop_size);

            // 处理该帧
            self.process_frame(&mut frame);

            if self.mode.uses_overlap() {
                // 重叠相加法：将上一帧保存的尾部与当前帧头部叠加
                let overlap_len = self.overlap_buffer.len();
                for i in 0..overlap_len {
                    frame[i] += self.overlap_buffer[i];
                }
                // 保存当前帧的尾部供下一帧使用
                self.overlap_buffer.copy_from_slice(&frame[hop_size..frame_size]);
                // 只输出帧移部分的样本
                output.extend_from_slice(&frame[..hop_size]);
            } else {
                // 无重叠：直接输出整帧
                output.extend_from_slice(&frame);
            }
        }

        output
    }

    /// 返回已处理的帧数
    pub fn frames_processed(&self) -> u64 {
        self.frames_processed
    }
}

// ============================================================
// 参考特征提取
// ============================================================

/// 从录音数据中构建参考特征向量集合
/// 对整段音频分帧提取特征，跳过静音帧，超过100帧时均匀下采样
pub fn build_reference_features(
    audio: &[f32],
    mode: ProcessingMode,
    actual_sample_rate: u32,
) -> Vec<Vec<f32>> {
    // 计算帧大小（与实时处理保持一致）
    let frame_size =
        (actual_sample_rate as f64 * mode.frame_duration_ms() as f64 / 1000.0).round() as usize;
    let frame_size = frame_size.max(64);
    let hop_size = frame_size; // 参考提取时不重叠
    let n_mels = 32;
    let mut planner = FftPlanner::new();
    let window = if mode.uses_window() {
        hanning_window(frame_size)
    } else {
        vec![1.0; frame_size]
    };

    let mut features = Vec::new();
    let mut pos = 0;

    // 逐帧提取特征
    while pos + frame_size <= audio.len() {
        let frame = &audio[pos..pos + frame_size];

        // 跳过静音帧
        let rms = (frame.iter().map(|&x| x * x).sum::<f32>() / frame.len() as f32).sqrt();
        if rms < 0.001 {
            pos += hop_size;
            continue;
        }

        // 根据模式提取特征
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

    // 如果特征帧数过多，均匀下采样到最多100帧
    let max_features = 100;
    if features.len() > max_features {
        let step = features.len() as f32 / max_features as f32;
        (0..max_features)
            .map(|i| features[(i as f32 * step) as usize].clone())
            .collect()
    } else {
        features
    }
}

// ============================================================
// 录音
// ============================================================

/// 从指定设备录制参考音频
/// duration_secs: 录音时长（秒）
/// 返回单声道（mono）音频样本
pub fn record_reference(
    device: &cpal::Device,
    config: &StreamConfig,
    duration_secs: u64,
) -> Result<Vec<f32>, String> {
    let sample_rate = config.sample_rate.0;
    let channels = config.channels as usize;
    // 预分配录音缓冲区
    let total_samples = sample_rate as usize * channels * duration_secs as usize;

    let recorded = Arc::new(Mutex::new(Vec::with_capacity(total_samples)));
    let recorded_clone = recorded.clone();

    // 创建输入音频流，回调中将数据追加到缓冲区
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

    // 开始录音，阻塞等待指定时长
    stream.play().map_err(|e| format!("无法开始录音: {}", e))?;
    std::thread::sleep(std::time::Duration::from_secs(duration_secs));
    drop(stream); // 停止录音

    // 提取录音数据
    let samples = Arc::try_unwrap(recorded).unwrap().into_inner().unwrap();

    // 多声道转单声道：取第一个声道
    let mono = if channels > 1 {
        samples.iter().step_by(channels).copied().collect()
    } else {
        samples
    };

    Ok(mono)
}

// ============================================================
// 音频流管理
// ============================================================

/// 音频会话：管理输入流、输出流和处理器
/// 启动后，输入流采集的音频经处理器处理后通过输出流播放
pub struct AudioSession {
    /// 输入音频流（麦克风采集）
    pub input_stream: Stream,
    /// 输出音频流（扬声器/虚拟线缆播放）
    pub output_stream: Stream,
    /// 音频处理器（线程安全共享）
    pub processor: Arc<Mutex<AudioProcessor>>,
}

impl AudioSession {
    /// 启动音频会话
    /// 创建输入流和输出流，建立从输入→处理→输出的实时音频管线
    pub fn start(
        input_device: &cpal::Device,
        output_device: &cpal::Device,
        mode: ProcessingMode,
        targets: Vec<TargetVoice>,
    ) -> Result<Self, String> {
        // 获取输入/输出设备的默认配置
        let default_input_config = input_device
            .default_input_config()
            .map_err(|e| format!("输入设备配置错误: {}", e))?;
        let default_output_config = output_device
            .default_output_config()
            .map_err(|e| format!("输出设备配置错误: {}", e))?;

        let input_sample_rate = default_input_config.sample_rate().0;
        let output_sample_rate = default_output_config.sample_rate().0;

        let input_config: StreamConfig = default_input_config.into();
        let output_config: StreamConfig = default_output_config.into();

        // 以输出采样率创建处理器
        let processor = Arc::new(Mutex::new(AudioProcessor::new(
            mode,
            targets,
            output_sample_rate,
        )));

        // 创建有界通道：输入回调将处理后的数据发送给输出回调
        // 缓冲区大小8，防止输出端积压过多数据导致延迟
        let (tx, rx) = std::sync::mpsc::sync_channel::<Vec<f32>>(8);

        // ---- 创建输入流 ----
        let processor_in = processor.clone();
        let channels_in = input_config.channels as usize;
        // 检测是否需要重采样（输入/输出采样率不同）
        let need_resample = input_sample_rate != output_sample_rate;
        let ratio = output_sample_rate as f64 / input_sample_rate as f64;

        let input_stream = input_device
            .build_input_stream(
                &input_config,
                move |data: &[f32], _: &cpal::InputCallbackInfo| {
                    // 多声道转单声道
                    let mono: Vec<f32> = if channels_in > 1 {
                        data.chunks(channels_in).map(|ch| ch[0]).collect()
                    } else {
                        data.to_vec()
                    };

                    // 如果采样率不匹配，进行线性插值重采样
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
                            // 线性插值
                            resampled.push(s0 + (s1 - s0) * frac as f32);
                        }
                        resampled
                    } else {
                        mono
                    };

                    // 送入处理器，将结果发送给输出端
                    let mut proc = processor_in.lock().unwrap();
                    let output = proc.process_samples(&samples);
                    if !output.is_empty() {
                        let _ = tx.try_send(output);
                    }
                },
                |err| eprintln!("输入流错误: {}", err),
                None,
            )
            .map_err(|e| format!("创建输入流失败: {}", e))?;

        // ---- 创建输出流 ----
        let channels_out = output_config.channels as usize;
        let sample_rate_out = output_config.sample_rate.0;
        // 限制输出缓冲区最大为100ms的数据，防止延迟累积
        let max_output_samples = sample_rate_out as usize * channels_out / 10;
        let mut output_buffer: VecDeque<f32> = VecDeque::with_capacity(max_output_samples);

        let output_stream = output_device
            .build_output_stream(
                &output_config,
                move |data: &mut [f32], _: &cpal::OutputCallbackInfo| {
                    // 从通道接收处理后的数据，追加到输出缓冲区
                    while let Ok(chunk) = rx.try_recv() {
                        for &s in &chunk {
                            output_buffer.push_back(s);
                        }
                    }

                    // 限制输出缓冲区大小，丢弃过旧的数据
                    while output_buffer.len() > max_output_samples {
                        output_buffer.pop_front();
                    }

                    // 填充输出数据：单声道扩展到多声道
                    for frame in data.chunks_mut(channels_out) {
                        let sample = output_buffer.pop_front().unwrap_or(0.0);
                        for ch in frame.iter_mut() {
                            *ch = sample;
                        }
                    }
                },
                |err| eprintln!("输出流错误: {}", err),
                None,
            )
            .map_err(|e| format!("创建输出流失败: {}", e))?;

        // 启动两个音频流
        input_stream.play().map_err(|e| format!("启动输入流失败: {}", e))?;
        output_stream.play().map_err(|e| format!("启动输出流失败: {}", e))?;

        Ok(AudioSession {
            input_stream,
            output_stream,
            processor,
        })
    }

    /// 停止音频会话，释放音频流资源
    pub fn stop(self) {
        drop(self.input_stream);
        drop(self.output_stream);
    }
}

// ============================================================
// 虚拟音频线缆检测
// ============================================================

/// 虚拟音频线缆设备名称中的关键词
/// 用于自动识别系统中安装的虚拟音频设备
const VIRTUAL_CABLE_KEYWORDS: &[&str] = &[
    "CABLE",       // VB-Cable
    "VB-Audio",    // VB-Audio 系列产品
    "Virtual",     // 通用虚拟设备
    "虚拟",         // 中文"虚拟"
    "VAC",         // Virtual Audio Cable
    "Voicemeeter", // Voicemeeter
    "VBAN",        // VBAN 协议设备
];

/// 判断设备名称是否匹配虚拟音频线缆
fn is_virtual_cable_name(name: &str) -> bool {
    let upper = name.to_uppercase();
    VIRTUAL_CABLE_KEYWORDS.iter().any(|kw| upper.contains(&kw.to_uppercase()))
}

/// 在输出设备列表中检测虚拟音频线缆，返回其名称
pub fn detect_virtual_cable_output() -> Option<String> {
    list_output_devices().into_iter().find(|n| is_virtual_cable_name(n))
}

/// 在输入设备列表中检测虚拟音频线缆，返回其名称
pub fn detect_virtual_cable_input() -> Option<String> {
    list_input_devices().into_iter().find(|n| is_virtual_cable_name(n))
}

/// 检测系统中是否存在虚拟音频线缆
pub fn has_virtual_cable() -> bool {
    detect_virtual_cable_output().is_some()
}

/// 在给定的输出设备列表中查找虚拟音频线缆的索引
pub fn find_virtual_cable_output_index(devices: &[String]) -> Option<usize> {
    devices.iter().position(|n| is_virtual_cable_name(n))
}

// ============================================================
// 设备枚举
// ============================================================

/// 列出所有可用的输入音频设备名称
/// 优先添加默认设备，然后遍历所有设备（去重）
pub fn list_input_devices() -> Vec<String> {
    let host = cpal::default_host();
    let mut devices = Vec::new();
    // 添加默认输入设备
    if let Some(input) = host.default_input_device() {
        devices.push(input.name().unwrap_or_default());
    }
    // 遍历所有输入设备，去重添加
    if let Ok(iter) = host.input_devices() {
        for d in iter {
            let name = d.name().unwrap_or_default();
            if !devices.contains(&name) {
                devices.push(name);
            }
        }
    }
    devices
}

/// 列出所有可用的输出音频设备名称
pub fn list_output_devices() -> Vec<String> {
    let host = cpal::default_host();
    let mut devices = Vec::new();
    // 添加默认输出设备
    if let Some(output) = host.default_output_device() {
        devices.push(output.name().unwrap_or_default());
    }
    // 遍历所有输出设备，去重添加
    if let Ok(iter) = host.output_devices() {
        for d in iter {
            let name = d.name().unwrap_or_default();
            if !devices.contains(&name) {
                devices.push(name);
            }
        }
    }
    devices
}

/// 根据名称查找输入设备
/// 先检查默认设备是否匹配，再遍历所有设备
pub fn find_input_device(name: &str) -> Option<cpal::Device> {
    let host = cpal::default_host();
    if let Some(default) = host.default_input_device() {
        if default.name().map(|n| n == name).unwrap_or(false) {
            return Some(default);
        }
    }
    if let Ok(iter) = host.input_devices() {
        for d in iter {
            if d.name().map(|n| n == name).unwrap_or(false) {
                return Some(d);
            }
        }
    }
    None
}

/// 根据名称查找输出设备
pub fn find_output_device(name: &str) -> Option<cpal::Device> {
    let host = cpal::default_host();
    if let Some(default) = host.default_output_device() {
        if default.name().map(|n| n == name).unwrap_or(false) {
            return Some(default);
        }
    }
    if let Ok(iter) = host.output_devices() {
        for d in iter {
            if d.name().map(|n| n == name).unwrap_or(false) {
                return Some(d);
            }
        }
    }
    None
}

// ============================================================
// 持久化
// ============================================================

/// 获取数据文件路径（与可执行文件同目录下的 vocal_sieve_data.json）
pub fn get_data_path() -> std::path::PathBuf {
    let mut path = std::env::current_exe().unwrap_or_else(|_| std::path::PathBuf::from("."));
    path.pop();
    path.push("vocal_sieve_data.json");
    path
}

/// 将目标人声列表保存到JSON文件
pub fn save_targets(targets: &[TargetVoice]) -> Result<(), String> {
    let path = get_data_path();
    let json = serde_json::to_string_pretty(targets).map_err(|e| e.to_string())?;
    std::fs::write(&path, json).map_err(|e| format!("保存失败: {}", e))
}

/// 从JSON文件加载目标人声列表
pub fn load_targets() -> Result<Vec<TargetVoice>, String> {
    let path = get_data_path();
    if !path.exists() {
        return Ok(Vec::new());
    }
    let json = std::fs::read_to_string(&path).map_err(|e| format!("读取失败: {}", e))?;
    serde_json::from_str(&json).map_err(|e| format!("解析失败: {}", e))
}
