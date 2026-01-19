%% fig3_style_sm_insaf_en.m
% Fig. 3 Style: SM-INSAF NMSD Comparison — Subband Number / Step Size / Past Weights

clear; clc; close all; rng(1);

%% Basic Parameters
fs = 8000; T = 5; L = fs*T; M = 512;
mu0 = 0.3; delta = 1e-6; t_bound = 2; beta = 0.9; SNRdB = 20;
P = 3; rho_list = [1, 0.6, 0.2]; N_list = [2, 4, 8]; P_list = [1, 2, 3];
step = 8;

%% Generate AR(1) Input and Desired Signal
w_noise = randn(L,1);
u = filter(1,[1 -0.9], w_noise);
w_true = fir1(M-1, 0.1)';
clean = filter(w_true,1,u);
noiseP = var(clean)/10^(SNRdB/10);
d = clean + sqrt(noiseP)*randn(L,1);

%% Fixed Filter Bank (for shared cases)
N_plot = 8;
Lp = 256;
proto = fir1(Lp-1,1/N_plot,hamming(Lp))*(2/sqrt(N_plot));
n = 0:(Lp-1);
h = zeros(N_plot, Lp);
for i = 0:N_plot-1
  theta = (2*i+1)*pi/(4*N_plot);
  h(i+1,:) = proto .* cos(theta*(2*n-(Lp-1)) + (-1)^i*pi/4);
end

%% Subband Decomposition (for fixed N=8)
u_sb = zeros(L,N_plot); d_sb = zeros(L,N_plot); nsb = zeros(L,N_plot);
noise = sqrt(noiseP)*randn(L,1);
for i = 1:N_plot
  u_sb(:,i) = filter(h(i,:),1,u);
  d_sb(:,i) = filter(h(i,:),1,d);
  nsb(:,i)  = filter(h(i,:),1,noise);
end
sigma_eta = var(nsb,0,1)'; 
gamma_i   = t_bound * sqrt(sigma_eta);

%% Time Axis
total_it = floor((L - M)/step);
x_axis = (1:total_it) * step / 1e4;   % ×10^4 input samples

%% ---------- Subplot (a): Different Number of Subbands ----------
nmsd_N = zeros(3,total_it);
for ni = 1:3
  Nsub = N_list(ni);
  proto = fir1(Lp-1,1/Nsub,hamming(Lp))*(2/sqrt(Nsub));
  h = zeros(Nsub,Lp); n = 0:(Lp-1);
  for i = 0:Nsub-1
    theta = (2*i+1)*pi/(4*Nsub);
    h(i+1,:) = proto .* cos(theta*(2*n-(Lp-1)) + (-1)^i*pi/4);
  end
  uD = zeros(L,Nsub); dD = zeros(L,Nsub); nsb = zeros(L,Nsub);
  for i = 1:Nsub
    uD(:,i) = filter(h(i,:),1,u);
    dD(:,i) = filter(h(i,:),1,d);
    nsb(:,i)= filter(h(i,:),1,noise);
  end
  gamma = t_bound * sqrt(var(nsb,0,1)');
  W = zeros(M,3); sigma_eps = zeros(Nsub,1);
  for k = 1:total_it
    idx = M + (k-1)*step;
    wbar = mean(W,2); dW = zeros(M,1);
    for i = 1:Nsub
      uvec = uD(idx:-1:idx-M+1,i);
      e = dD(idx,i) - wbar'*uvec;
      sigma_eps(i) = beta*sigma_eps(i) + (1-beta)*abs(e);
      if min(abs(e),sigma_eps(i)) > gamma(i)
        mu = 1 - gamma(i)/max(sigma_eps(i),1e-12);
        dW = dW + (mu0*mu*e*uvec)/(uvec'*uvec + delta);
      end
    end
    w = wbar + dW; W = [w, W(:,1:2)];
    nmsd_N(ni,k) = 10*log10(norm(w_true - w)^2 / norm(w_true)^2);
  end
end

%% ---------- Subplot (b): Different Step Sizes ----------
step_list = [4, 8, 16];
nmsd_step = zeros(3, total_it);
min_len = inf;

for si = 1:3
  step_now = step_list(si);
  total_now = floor((L - M)/step_now);
  W = zeros(M,3); sigma_eps = zeros(N_plot,1);
  tmp_nmsd = zeros(1,total_now);

  for k = 1:total_now
    idx = M + (k-1)*step_now;
    wbar = mean(W,2); dW = zeros(M,1);
    for i = 1:N_plot
      uvec = u_sb(idx:-1:idx-M+1,i);
      e = d_sb(idx,i) - wbar'*uvec;
      sigma_eps(i) = beta*sigma_eps(i) + (1-beta)*abs(e);
      if min(abs(e),sigma_eps(i)) > gamma_i(i)
        mu = 1 - gamma_i(i)/max(sigma_eps(i),1e-12);
        dW = dW + (mu0*mu*e*uvec)/(uvec'*uvec + delta);
      end
    end
    w = wbar + dW; W = [w, W(:,1:2)];
    tmp_nmsd(k) = 10*log10(norm(w_true - w)^2 / norm(w_true)^2);
  end
  min_len = min(min_len, numel(tmp_nmsd));
  nmsd_step(si,1:numel(tmp_nmsd)) = tmp_nmsd;
end
x_axis_step = (1:min_len) * step_list(1) / 1e4;

%% ---------- Subplot (c): Different Number of Reused Weight Vectors ----------
nmsd_P = zeros(3,total_it);
for pi = 1:3
  P_now = P_list(pi);
  W = zeros(M,P_now); sigma_eps = zeros(N_plot,1);
  for k = 1:total_it
    idx = M + (k-1)*step;
    wbar = mean(W,2); dW = zeros(M,1);
    for i = 1:N_plot
      uvec = u_sb(idx:-1:idx-M+1,i);
      e = d_sb(idx,i) - wbar'*uvec;
      sigma_eps(i) = beta*sigma_eps(i) + (1-beta)*abs(e);
      if min(abs(e),sigma_eps(i)) > gamma_i(i)
        mu = 1 - gamma_i(i)/max(sigma_eps(i),1e-12);
        dW = dW + (mu0*mu*e*uvec)/(uvec'*uvec + delta);
      end
    end
    w = wbar + dW; W = [w, W(:,1:end-1)];
    nmsd_P(pi,k) = 10*log10(norm(w_true - w)^2 / norm(w_true)^2);
  end
end

%% ---------- Plotting ----------
figure('Position',[100 100 880 300]);

% Subplot (a)
subplot(1,3,1);
plot(x_axis,nmsd_N(1,:),'g--','LineWidth',1.4); hold on;
plot(x_axis,nmsd_N(2,:),'b-.','LineWidth',1.4);
plot(x_axis,nmsd_N(3,:),'r-','LineWidth',1.4);
title('{\it(a)} Number of Subbands');
legend('N=2','N=4','N=8','Location','northeast');
xlabel('Input Samples $\times 10^4$','Interpreter','latex');
ylabel('NMSD (dB)');
grid on; ylim([-22 0]);

% Subplot (b)
subplot(1,3,2);
plot(x_axis_step,nmsd_step(1,1:min_len),'r-','LineWidth',1.4); hold on;
plot(x_axis_step,nmsd_step(2,1:min_len),'b-.','LineWidth',1.4);
plot(x_axis_step,nmsd_step(3,1:min_len),'g--','LineWidth',1.4);
title('{\it(b)} Step Size');
legend('step=4','step=8','step=16','Location','northeast');
xlabel('Input Samples $\times 10^4$','Interpreter','latex');
grid on; ylim([-22 0]);

% Subplot (c)
subplot(1,3,3);
plot(x_axis,nmsd_P(1,:),'r-','LineWidth',1.4); hold on;
plot(x_axis,nmsd_P(2,:),'b-.','LineWidth',1.4);
plot(x_axis,nmsd_P(3,:),'g--','LineWidth',1.4);
title('{\it(c)} Number of Weight Vectors');
legend('P=1','P=2','P=3','Location','northeast');
xlabel('Input Samples $\times 10^4$','Interpreter','latex');
grid on; ylim([-22 0]);

% Font and Export Settings
set(findall(gcf,'-property','FontName'),'FontName','Times New Roman');
sgtitle('Performance Comparison of SM-INSAF under Different Parameters');
