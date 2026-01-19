%% 1) 参数设置
fs     = 8000;        % 原始采样率 (Hz)
T      = 5;           % 总时长 (s)
L      = fs * T;      % 样本数
a      = 0.9;         % AR(1) 系数
sigma  = 1;           % 白噪声标准差

N      = 8;           % 子带数
Lh     = 64;          % 原型滤波器长度 (偶数)
Fc     = 1/N;         % 原型低通归一化截止频率 (Nyquist=1)

%% 2) 生成 AR(1) 信号： x[n] = a*x[n-1] + noise
noise = sigma * randn(L,1);
x     = filter(1, [1, -a], noise);

%% 3) 设计原型低通 FIR 滤波器
h_proto = fir1(Lh-1, Fc, hamming(Lh));  

%% 4) 生成 N 支分析滤波器 (余弦调制)
%    hi[n] = 2*h_proto[n] * cos( (2*i+1)*pi/(2N)*(n-(Lh-1)/2) )
n = 0:Lh-1;
hi = zeros(N, Lh);
for i = 0:N-1
    hi(i+1, :) = 2 * h_proto .* ...
        cos((2*i+1) * pi/(2*N) * (n - (Lh-1)/2));
end

%% 5) 对原始信号做子带滤波 + N 倍降采样
%    yi = hi * x ； xi = downsample(yi, N)
x_sub = cell(N,1);
for i = 1:N
    y = filter(hi(i,:), 1, x);   % 子带 i 的滤波输出
    x_sub{i} = y(i:N:end);       % 降采样，保留第 i 个样本开始、步长 N
end

%% 6) 可视化：比较原始信号和每个子带信号的自相关
maxLag = 20;
figure;
subplot(2,1,1);
[rx, lags] = xcorr(x, maxLag, 'coeff');
stem(lags, rx, 'filled');
title('原始 AR(1) 信号 自相关');
xlabel('滞后'); ylabel('归一化自相关');

subplot(2,1,2);
hold on;
colors = lines(N);
for i = 1:N
    [r_sub, l] = xcorr(x_sub{i}, maxLag, 'coeff');
    stem(l, r_sub, 'marker','none','Color',colors(i,:),...
         'DisplayName',['子带 ',num2str(i)]);
end
hold off;
legend('show');
title('各子带信号 自相关（降采样后）');
xlabel('滞后'); ylabel('归一化自相关');

%% 7) 验证“白化”效果
% 可以看到，原始信号在滞后1处自相关较大，而各子带信号
% 降采样后相邻样本（等同滞后1）的自相关显著减小，近似白化。
