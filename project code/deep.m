%% demo_ssm_insaf_nodown.m
% SSM-INSAF final version without subband downsampling
clear; clc; close all; rng(1);

% 1. Global parameters
fs      = 8000;       % sample rate (Hz)
T       = 5;          % duration (s)
L       = fs * T;     % number of samples
M       = 512;        % filter length
Nsub    = 16;         % number of subbands
Lp      = 256;        % prototype length
mu0     = 0.3;        % global step size
delta   = 1e-6;       % regularization constant
t_bound = 2;          % threshold factor
P       = 2;          % number of past weights to reuse
beta    = 0.9;        % smoothing factor
SNRdB   = 20;         % SNR in dB

% 2. Generate AR(1) input, echo path, and noisy desired
w_noise = randn(L,1);
% u       = filter(1,[1 -0.9], w_noise);
 u = 0.1*randn(L,1);


w_true  = fir1(M-1,0.1)';        % true echo path
clean   = filter(w_true,1,u);
noiseP  = var(clean)/10^(SNRdB/10);
d       = clean + sqrt(noiseP)*randn(L,1);

% 3. Design paraunitary cosine-modulated filterbank
proto = fir1(Lp-1,1/Nsub,hamming(Lp)) * (2/sqrt(Nsub));
h     = zeros(Nsub, Lp);
for i = 0:(Nsub-1)
  for n = 0:(Lp-1)
    h(i+1,n+1) = 2 * proto(n+1) * ...
      cos((2*i+1)*pi/(4*Nsub)*(2*n-(Lp-1)) + (-1)^i*pi/4);
  end
end

% 4. Subband decomposition (no downsampling)
u_sb = zeros(L, Nsub);
d_sb = zeros(L, Nsub);
noise = sqrt(noiseP)*randn(L,1);
nsb   = zeros(L, Nsub);
for i = 1:Nsub
  u_sb(:,i) = filter(h(i,:),1,u);
  d_sb(:,i) = filter(h(i,:),1,d);
  nsb(:,i)  = filter(h(i,:),1,noise);
end

% 5. Compute per-subband noise variance and thresholds
sigma_eta = var(nsb,0,1)';      % Nsub×1
gamma_i   = t_bound * sigma_eta;

% 6. Initialization
W_past    = zeros(M, P);        % store past P weight vectors
sigma_eps = zeros(Nsub,1);      % smoothed abs-errors
H_store   = zeros(M, L-M+1);    % to save each w
upd_cnt   = zeros(Nsub,1);      % update counters

% 7. SSM-INSAF main loop
for k = M : L
  idx  = k - M + 1;
  wbar = mean(W_past,2);
  dW   = zeros(M,1);
  for i = 1:Nsub
    uvec           = u_sb(k:-1:k-M+1, i);
    e_i            = d_sb(k,        i) - uvec' * wbar;
    sigma_eps(i)   = beta * sigma_eps(i) + (1-beta) * abs(e_i);
    if min(abs(e_i), sigma_eps(i)) > gamma_i(i)
      mu_i = 1 - gamma_i(i) / sigma_eps(i);
      dW   = dW + (mu0 * mu_i * e_i * uvec) / (uvec'*uvec + delta);
      upd_cnt(i) = upd_cnt(i) + 1;
    end
  end
  w_new       = wbar + dW;
  W_past      = [w_new, W_past(:,1:end-1)];  % rotate history
  H_store(:,idx) = w_new;
end

% 8. Display average update rate
total_it  = L - M + 1;
avg_rate  = mean(upd_cnt/total_it) * 100;
fprintf('Average update rate: %.2f%%\n', avg_rate);

% 9. Compute and plot NMSD
nmsd = 10*log10( sum((H_store - w_true).^2,1) / (norm(w_true)^2 + eps) );
figure;
plot(1:total_it, nmsd, 'LineWidth',1.4);
xlabel('Iteration');
ylabel('NMSD (dB)');
title(sprintf('SSM-INSAF No Downsampling (M=%d, Nsub=%d)', M, Nsub));
grid on; ylim([-50,5]);
