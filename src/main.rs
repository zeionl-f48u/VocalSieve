// ============================================================
// VocalSieve - 实时人声筛选器 · 主程序入口与GUI
// ============================================================
// 本文件实现：
//   - eframe/egui GUI主窗口
//   - 应用状态管理（空闲/录音/运行）
//   - 音频设备选择与刷新
//   - 处理模式切换
//   - 目标人声管理（添加/删除/持久化）
//   - 虚拟音频线缆检测与配置
//   - 实时状态显示（相似度、增益）
//   - 中文字体加载
// ============================================================

mod audio;

use audio::{
    find_input_device, find_output_device, list_input_devices, list_output_devices,
    load_targets, record_reference, save_targets, build_reference_features,
    AudioSession, ProcessingMode, TargetAction, TargetVoice,
    detect_virtual_cable_output, find_virtual_cable_output_index,
};
use cpal::traits::DeviceTrait;
use eframe::egui;

// ============================================================
// 应用状态
// ============================================================

/// 应用状态枚举，表示当前所处的阶段
enum AppState {
    /// 空闲状态：未在录音或处理
    Idle,
    /// 录音状态：正在录制参考声音
    /// countdown: 倒计时计数器（每100ms减1，50次=5秒）
    /// target_name: 目标人声名称
    /// target_action: 对目标执行的动作（消除/增强）
    Recording { countdown: u32, target_name: String, target_action: TargetAction },
    /// 运行状态：实时音频处理已启动
    Running,
}

/// 主应用结构体，持有所有GUI状态和音频会话
struct VocalSieveApp {
    // --- 设备相关 ---
    /// 可用的输入设备名称列表
    input_devices: Vec<String>,
    /// 可用的输出设备名称列表
    output_devices: Vec<String>,
    /// 当前选中的输入设备索引
    selected_input: usize,
    /// 当前选中的输出设备索引
    selected_output: usize,

    // --- 模式相关 ---
    /// 当前选中的处理模式索引（对应 ProcessingMode::all()）
    selected_mode: usize,

    // --- 目标人声相关 ---
    /// 已添加的目标人声列表
    targets: Vec<TargetVoice>,
    /// 新目标名称的输入框内容
    new_target_name: String,
    /// 新目标动作选择：0=消除，1=增强
    new_target_action: usize,

    // --- 运行状态 ---
    /// 当前应用状态
    state: AppState,
    /// 活跃的音频会话（包含输入/输出流和处理器）
    session: Option<AudioSession>,

    // --- 录音线程 ---
    /// 录音线程的JoinHandle，用于检测录音是否完成并获取结果
    /// 线程返回 Result<Vec<Vec<f32>>, String>：成功返回特征向量集合，失败返回错误信息
    recording_handle: Option<std::thread::JoinHandle<Result<Vec<Vec<f32>>, String>>>,

    // --- 状态信息 ---
    /// 底部状态栏显示的消息
    status_msg: String,

    // --- 实时监控数据 ---
    /// 各目标人声的最新相似度值（从处理器读取）
    live_similarities: Vec<f32>,
    /// 当前应用的增益值（从处理器读取）
    live_gain: f32,
}

impl VocalSieveApp {
    /// 创建应用实例，初始化设备列表和已保存的目标
    fn new() -> Self {
        // 枚举系统中的音频设备
        let input_devices = list_input_devices();
        let output_devices = list_output_devices();
        // 从磁盘加载之前保存的目标人声
        let targets = load_targets().unwrap_or_default();

        VocalSieveApp {
            // 默认选择第一个输入设备
            selected_input: 0,
            // 如果有多个输出设备，默认选第二个（通常是虚拟线缆或非默认扬声器）
            selected_output: if output_devices.len() > 1 { 1 } else { 0 },
            // 默认选择平衡模式（索引1）
            selected_mode: 1,
            input_devices,
            output_devices,
            targets,
            new_target_name: String::new(),
            new_target_action: 0,
            state: AppState::Idle,
            session: None,
            recording_handle: None,
            status_msg: String::new(),
            live_similarities: Vec::new(),
            live_gain: 1.0,
        }
    }

    /// 根据当前选中的模式索引获取 ProcessingMode 枚举值
    fn current_mode(&self) -> ProcessingMode {
        ProcessingMode::all()[self.selected_mode]
    }

    /// 刷新音频设备列表，并修正越界的选中索引
    fn refresh_devices(&mut self) {
        self.input_devices = list_input_devices();
        self.output_devices = list_output_devices();
        // 防止选中索引超出新列表范围
        if self.selected_input >= self.input_devices.len() {
            self.selected_input = 0;
        }
        if self.selected_output >= self.output_devices.len() {
            self.selected_output = 0;
        }
    }

    /// 开始录制参考声音
    /// 在新线程中执行5秒录音，录音完成后自动提取特征并添加到目标列表
    fn start_recording(&mut self) {
        let input_name = self.input_devices.get(self.selected_input).cloned();
        let mode = self.current_mode();
        let name = self.new_target_name.clone();
        let action = if self.new_target_action == 0 {
            TargetAction::Suppress
        } else {
            TargetAction::Enhance
        };

        // 验证：名称不能为空
        if name.is_empty() {
            self.status_msg = "请输入目标名称".into();
            return;
        }

        // 验证：检查当前模式的消除目标数量限制
        let max = mode.max_targets();
        let current_suppress = self.targets.iter().filter(|t| t.action == TargetAction::Suppress).count();
        let _current_enhance = self.targets.iter().filter(|t| t.action == TargetAction::Enhance).count();
        if action == TargetAction::Suppress && current_suppress >= max {
            self.status_msg = format!("该模式最多支持 {} 个消除目标", max);
            return;
        }

        if let Some(input_name) = input_name {
            self.status_msg = "正在录音... 请说话（5秒）".into();
            // 切换到录音状态，启动倒计时
            self.state = AppState::Recording {
                countdown: 50, // 50 × 100ms = 5秒
                target_name: name,
                target_action: action,
            };

            // 在新线程中执行录音和特征提取，避免阻塞GUI
            let handle = std::thread::spawn(move || {
                // 查找输入设备
                let device = find_input_device(&input_name)
                    .ok_or("找不到输入设备")?;
                // 获取设备默认配置
                let config = device.default_input_config()
                    .map_err(|e| format!("配置错误: {}", e))?;
                let stream_config: cpal::StreamConfig = config.into();
                // 录制5秒参考音频
                let audio = record_reference(&device, &stream_config, 5)?;
                // 从参考音频中提取特征向量集合
                let features = build_reference_features(&audio, mode, stream_config.sample_rate.0);
                Ok(features)
            });
            self.recording_handle = Some(handle);
        }
    }

    /// 启动实时音频处理
    /// 根据当前选择的设备、模式和目标创建 AudioSession
    fn start_processing(&mut self) {
        // 验证：至少需要一个目标
        if self.targets.is_empty() {
            self.status_msg = "请先添加至少一个目标".into();
            return;
        }

        let input_name = self.input_devices.get(self.selected_input).cloned();
        let output_name = self.output_devices.get(self.selected_output).cloned();
        let mode = self.current_mode();
        let targets = self.targets.clone();

        if let (Some(input_name), Some(output_name)) = (input_name, output_name) {
            // 查找输入和输出设备
            match (
                find_input_device(&input_name),
                find_output_device(&output_name),
            ) {
                (Some(input_dev), Some(output_dev)) => {
                    // 创建并启动音频会话
                    match AudioSession::start(&input_dev, &output_dev, mode, targets) {
                        Ok(session) => {
                            self.session = Some(session);
                            self.state = AppState::Running;
                            self.status_msg = "实时处理已启动".into();
                        }
                        Err(e) => {
                            self.status_msg = format!("启动失败: {}", e);
                        }
                    }
                }
                _ => {
                    self.status_msg = "找不到所选设备".into();
                }
            }
        }
    }

    /// 停止实时音频处理
    /// 停止音频流并重置状态
    fn stop_processing(&mut self) {
        // 停止并丢弃音频会话（Drop时会自动停止流）
        if let Some(session) = self.session.take() {
            session.stop();
        }
        self.state = AppState::Idle;
        self.status_msg = "已停止".into();
        // 清空实时监控数据
        self.live_similarities.clear();
        self.live_gain = 1.0;
    }
}

// ============================================================
// eframe GUI 实现
// ============================================================

impl eframe::App for VocalSieveApp {
    /// 每帧调用一次，更新应用状态并绘制GUI
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // --- 检查录音线程是否完成 ---
        // 使用 take() 取出 JoinHandle，如果线程未完成则放回
        if let Some(handle) = self.recording_handle.take() {
            if !handle.is_finished() {
                // 线程仍在运行，放回等待下次检查
                self.recording_handle = Some(handle);
            } else {
                // 线程已完成，获取录音结果
                match handle.join() {
                    Ok(Ok(features)) => {
                        // 录音成功，从当前状态获取目标信息
                        let (name, action) = match &self.state {
                            AppState::Recording { target_name, target_action, .. } => {
                                (target_name.clone(), *target_action)
                            }
                            _ => (String::new(), TargetAction::Suppress),
                        };
                        if !name.is_empty() {
                            // 创建新目标并添加到列表
                            self.targets.push(TargetVoice {
                                name,
                                action,
                                reference_features: features,
                            });
                            // 持久化保存到磁盘
                            let _ = save_targets(&self.targets);
                            self.status_msg = format!("已添加目标，共 {} 个", self.targets.len());
                        }
                        self.state = AppState::Idle;
                    }
                    Ok(Err(e)) => {
                        // 录音过程中出错
                        self.status_msg = format!("录音失败: {}", e);
                        self.state = AppState::Idle;
                    }
                    Err(_) => {
                        // 录音线程panic
                        self.status_msg = "录音线程异常".into();
                        self.state = AppState::Idle;
                    }
                }
            }
        }

        // --- 更新录音倒计时 ---
        if let AppState::Recording { countdown, .. } = &mut self.state {
            // 每帧减少1（约100ms一帧）
            *countdown = countdown.saturating_sub(1);
            // 倒计时结束且录音线程已返回结果
            if *countdown == 0 && self.recording_handle.is_none() {
                self.state = AppState::Idle;
            }
        }

        // --- 更新实时监控数据 ---
        // 从音频处理器中读取最新的相似度和增益值
        if let Some(session) = &self.session {
            if let Ok(proc) = session.processor.lock() {
                self.live_similarities = proc.last_similarities.clone();
                self.live_gain = proc.last_gain;
            }
        }

        // 持续请求重绘，确保UI实时更新
        ctx.request_repaint();

        // --- 绘制主界面 ---
        egui::CentralPanel::default().show(ctx, |ui| {
            // 标题
            ui.vertical_centered(|ui| {
                ui.add_space(8.0);
                ui.heading("VocalSieve - 实时人声筛选器");
                ui.add_space(4.0);
            });

            ui.separator();

            // === 设备选择面板 ===
            ui.collapsing("音频设备", |ui| {
                // 输入设备下拉框
                ui.horizontal(|ui| {
                    ui.label("输入设备:");
                    egui::ComboBox::from_id_salt("input_device")
                        .selected_text(
                            self.input_devices
                                .get(self.selected_input)
                                .map(|s| s.as_str())
                                .unwrap_or("(无)"),
                        )
                        .show_ui(ui, |ui| {
                            for (i, name) in self.input_devices.iter().enumerate() {
                                ui.selectable_value(&mut self.selected_input, i, name);
                            }
                        });
                });

                // 输出设备下拉框
                ui.horizontal(|ui| {
                    ui.label("输出设备:");
                    egui::ComboBox::from_id_salt("output_device")
                        .selected_text(
                            self.output_devices
                                .get(self.selected_output)
                                .map(|s| s.as_str())
                                .unwrap_or("(无)"),
                        )
                        .show_ui(ui, |ui| {
                            for (i, name) in self.output_devices.iter().enumerate() {
                                ui.selectable_value(&mut self.selected_output, i, name);
                            }
                        });
                });

                // 刷新设备列表按钮
                ui.horizontal(|ui| {
                    if ui.button("刷新设备列表").clicked() {
                        self.refresh_devices();
                    }
                });

                ui.separator();

                // === 虚拟音频线缆配置 ===
                // 检测系统中是否安装了虚拟音频线缆（如VB-Cable），
                // 用于将处理后的音频输出到游戏中
                ui.label(egui::RichText::new("游戏输出配置").strong());
                ui.add_space(4.0);

                let cable_output = detect_virtual_cable_output();
                if let Some(cable_name) = &cable_output {
                    // 已检测到虚拟线缆：显示配置引导
                    ui.colored_label(egui::Color32::GREEN, format!("已检测到虚拟线缆: {}", cable_name));

                    // 一键将输出设备切换为虚拟线缆
                    ui.horizontal(|ui| {
                        if ui.button("一键配置为输出设备").clicked() {
                            if let Some(idx) = find_virtual_cable_output_index(&self.output_devices) {
                                self.selected_output = idx;
                                self.status_msg = format!("已将输出设备切换为: {}", cable_name);
                            }
                        }
                    });

                    // 使用说明
                    ui.add_space(4.0);
                    ui.label(
                        egui::RichText::new("使用方法:")
                            .small()
                            .color(egui::Color32::YELLOW),
                    );
                    ui.label(
                        egui::RichText::new("1. 点击上方按钮将输出设为虚拟线缆")
                            .small(),
                    );
                    ui.label(
                        egui::RichText::new("2. 在游戏中将麦克风/输入设备选为虚拟线缆")
                            .small(),
                    );
                    ui.label(
                        egui::RichText::new("3. 点击「开始实时处理」即可")
                            .small(),
                    );
                } else {
                    // 未检测到虚拟线缆：提供下载引导
                    ui.colored_label(egui::Color32::from_rgb(255, 150, 50), "未检测到虚拟音频线缆");
                    ui.add_space(4.0);
                    ui.label(
                        egui::RichText::new("要让游戏使用本软件的输出，需要安装虚拟音频线缆:")
                            .small()
                            .color(egui::Color32::YELLOW),
                    );
                    ui.label(
                        egui::RichText::new("推荐: VB-Cable (免费)")
                            .small(),
                    );
                    ui.horizontal(|ui| {
                        if ui.button("打开 VB-Cable 下载页").clicked() {
                            let _ = open::that("https://vb-audio.com/Cable/index.htm");
                        }
                    });
                    ui.label(
                        egui::RichText::new("安装后重启本软件即可自动检测")
                            .small()
                            .color(egui::Color32::GRAY),
                    );
                }
            });

            // === 处理模式面板 ===
            ui.collapsing("处理模式", |ui| {
                // 单选按钮组：性能/平衡/深度
                for (i, mode) in ProcessingMode::all().iter().enumerate() {
                    ui.radio_value(&mut self.selected_mode, i, mode.label());
                }
                // 显示当前模式的参数摘要
                let mode = self.current_mode();
                ui.label(format!(
                    "阈值: {:.2} | 消除增益: {:.2} | 增强增益: {:.1}x",
                    mode.similarity_threshold(),
                    mode.suppress_gain(),
                    mode.enhance_gain(),
                ));
            });

            // === 目标人声管理面板 ===
            ui.collapsing("目标人声管理", |ui| {
                // 已有目标列表
                if !self.targets.is_empty() {
                    ui.label(egui::RichText::new("已有目标:").strong());
                    // 收集待删除目标的索引（不能在迭代中修改列表）
                    let mut to_remove: Option<usize> = None;
                    for (i, target) in self.targets.iter().enumerate() {
                        ui.horizontal(|ui| {
                            // 根据动作类型显示不同颜色的标签
                            let action_label = match target.action {
                                TargetAction::Suppress => "消除",
                                TargetAction::Enhance => "增强",
                            };
                            let color = match target.action {
                                TargetAction::Suppress => egui::Color32::RED,
                                TargetAction::Enhance => egui::Color32::GREEN,
                            };
                            ui.colored_label(color, format!("{} [{}]", target.name, action_label));
                            // 显示参考特征帧数
                            ui.label(format!("({} 帧参考)", target.reference_features.len()));
                            // 删除按钮
                            if ui.small_button("删除").clicked() {
                                to_remove = Some(i);
                            }
                        });
                    }
                    // 执行删除并保存
                    if let Some(idx) = to_remove {
                        self.targets.remove(idx);
                        let _ = save_targets(&self.targets);
                    }
                    ui.separator();
                }

                // 添加新目标的表单
                let is_recording = matches!(self.state, AppState::Recording { .. });
                // 目标名称输入框（录音时禁用）
                ui.add_enabled(!is_recording, egui::TextEdit::singleline(&mut self.new_target_name).hint_text("目标名称"));
                // 动作选择：消除/增强
                ui.horizontal(|ui| {
                    ui.radio_value(&mut self.new_target_action, 0, "消除");
                    ui.radio_value(&mut self.new_target_action, 1, "增强");
                });
                // 录音按钮（录音时禁用，并显示倒计时）
                ui.add_enabled_ui(!is_recording, |ui| {
                    let label = if is_recording {
                        if let AppState::Recording { countdown, .. } = &self.state {
                            format!("录音中... {}s", (countdown + 9) / 10)
                        } else {
                            "录音中...".into()
                        }
                    } else {
                        "录制参考声音 (5秒)".into()
                    };
                    if ui.button(label).clicked() {
                        self.start_recording();
                    }
                });
            });

            ui.separator();

            // === 启动/停止按钮 ===
            ui.horizontal(|ui| {
                let is_running = matches!(self.state, AppState::Running);
                if is_running {
                    // 运行中：显示停止按钮
                    if ui.button("停止处理").clicked() {
                        self.stop_processing();
                    }
                } else {
                    // 未运行：显示启动按钮（无目标时禁用）
                    ui.add_enabled_ui(!self.targets.is_empty(), |ui| {
                        if ui.button("开始实时处理").clicked() {
                            self.start_processing();
                        }
                    });
                }
            });

            // === 实时状态面板 ===
            // 仅在运行状态下显示
            if matches!(self.state, AppState::Running) {
                ui.separator();
                ui.label(egui::RichText::new("实时状态").strong());

                // 增益指示器：根据增益值显示不同颜色
                // 红色=消除中（增益≈0），绿色=增强中（增益>1），白色=正常
                let gain_color = if self.live_gain < 0.1 {
                    egui::Color32::RED
                } else if self.live_gain > 1.5 {
                    egui::Color32::GREEN
                } else {
                    egui::Color32::WHITE
                };
                ui.colored_label(gain_color, format!("当前增益: {:.3}", self.live_gain));

                // 各目标的相似度显示
                for (i, target) in self.targets.iter().enumerate() {
                    let sim = self.live_similarities.get(i).copied().unwrap_or(0.0);
                    let threshold = self.current_mode().similarity_threshold();
                    // 超过阈值显示红色（匹配中），未超过显示绿色（安全）
                    let color = if sim >= threshold {
                        egui::Color32::from_rgb(255, 100, 100)
                    } else {
                        egui::Color32::from_rgb(100, 200, 100)
                    };
                    ui.colored_label(color, format!("{}: 相似度 {:.3} (阈值 {:.2})", target.name, sim, threshold));

                    // 相似度进度条，超过阈值时填充红色
                    let progress = (sim / 1.0).clamp(0.0, 1.0);
                    let mut bar = egui::ProgressBar::new(progress)
                        .text(format!("{:.0}%", sim * 100.0));
                    if sim >= threshold {
                        bar = bar.fill(egui::Color32::RED);
                    }
                    ui.add(bar);
                }
            }

            // === 状态消息 ===
            if !self.status_msg.is_empty() {
                ui.separator();
                ui.label(&self.status_msg);
            }
        });
    }
}

// ============================================================
// 程序入口
// ============================================================

/// 程序入口函数
/// 配置窗口属性、加载中文字体、启动eframe事件循环
fn main() -> eframe::Result<()> {
    // 配置原生窗口选项
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([480.0, 640.0])       // 默认窗口大小
            .with_min_inner_size([400.0, 500.0])    // 最小窗口大小
            .with_title("VocalSieve"),               // 窗口标题
        ..Default::default()
    };

    // 启动eframe原生运行时
    eframe::run_native(
        "VocalSieve",
        options,
        Box::new(|cc| {
            // 加载Windows系统中文字体（微软雅黑），解决中文显示为方框的问题
            let mut fonts = egui::FontDefinitions::default();
            if let Ok(font_data) = std::fs::read("C:\\Windows\\Fonts\\msyh.ttc") {
                // 将字体数据注册到egui字体系统
                fonts.font_data.insert(
                    "msyh".into(),
                    egui::FontData::from_owned(font_data),
                );
                // 将中文字体插入到 Proportional（比例字体）和 Monospace（等宽字体）族的首位
                // 首位优先级最高，确保中文字符优先使用此字体渲染
                fonts.families.entry(egui::FontFamily::Proportional).or_default().insert(0, "msyh".into());
                fonts.families.entry(egui::FontFamily::Monospace).or_default().insert(0, "msyh".into());
            }
            cc.egui_ctx.set_fonts(fonts);
            // 创建并返回应用实例
            Ok(Box::new(VocalSieveApp::new()))
        }),
    )
}
