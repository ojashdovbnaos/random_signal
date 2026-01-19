%% demo_ssm_insaf_snr.m
% Step2: SSM-INSAF convergence vs. SNR (M=64, Nsub=8, P=2, beta=0.9)
clear; clc; close all; rng(1);

% 1. Global parameters
fs       = 8000;    
T        = 5;       
L        = fs * T;  
M        = 64;      
mu0      = 0.1;     
delta    = 1e-3;    

% 2. Subband parameters
Nsub     = 8;       
Lp       = 64;      

% 3. Generate white input and true system
x        = 0.1 * randn(L,1);
b        = fir1(M-1, 0.1);

% 4. Design paraunitary CMFB
proto    = fir1(Lp-1, 1/Nsub, hamming(Lp)) * (2/sqrt(Nsub));
h        = zeros(Nsub, Lp);
for i = 0:(Nsub-1)
  for n = 0:(Lp-1)
    h(i+1,n+1) = 2 * proto(n+1) * ...
      cos((2*i+1)*pi/(4*Nsub)*(2*n-(Lp-1)) + (-1)^i*pi/4);
  end
end

% 5. Decompose x (no downsampling)
u_sb     = zeros(L, Nsub);
for i = 1:Nsub
  u_sb(:,i) = filter(h(i,:), 1, x);
end

% 6. SNR list & colors
SNRdB_list = [0, 10, 20, 40];
colors      = lines(numel(SNRdB_list));

figure; hold on;
for idxS = 1:numel(SNRdB_list)
  SNRdB    = SNRdB_list(idxS);
  
  % 6.1 clean + noisy d
  clean    = filter(b,1,x);
  noiseP   = var(clean)/10^(SNRdB/10);
  d        = clean + sqrt(noiseP) * randn(L,1);

  % 6.2 decompose d
  d_sb     = zeros(L, Nsub);
  for i = 1:Nsub
    d_sb(:,i) = filter(h(i,:), 1, d);
  end

  % 6.3 estimate subband noise variance & thresholds
  noise    = sqrt(noiseP) * randn(L,1);
  n_sb     = zeros(L, Nsub);
  for i = 1:Nsub
    n_sb(:,i) = filter(h(i,:), 1, noise);
  end
  sigma_eta = var(n_sb,0,1)';      
  t_bound   = 2;
  gamma_i   = t_bound * sigma_eta; 

  % 7. Initialization for SSM-INSAF
  P         = 2;
  beta      = 0.9;
  W_past    = zeros(M, P);       % columns hold past estimates
  sigma_eps = zeros(Nsub,1);     % smoothed |e_i|
  H_store   = zeros(M, L-M+1);

  % 8. SSM-INSAF main loop
  for k = M : L
    pos   = k - M + 1;
    wbar  = mean(W_past, 2);      
    dW    = zeros(M,1);
    for i = 1:Nsub
      uvec     = u_sb(k:-1:k-M+1,i);
      e_i      = d_sb(k,i) - uvec' * wbar;
      % smooth |e_i|
      sigma_eps(i) = beta * sigma_eps(i) + (1-beta) * abs(e_i);
      if min(abs(e_i), sigma_eps(i)) > gamma_i(i)
        mu_i = 1 - gamma_i(i) / sigma_eps(i);
        dW   = dW + (mu0 * mu_i * e_i * uvec) / (uvec'*uvec + delta);
      end
    end
    w_new        = wbar + dW;
    % rotate history
    W_past       = [w_new, W_past(:,1:end-1)];
    H_store(:,pos) = w_new;
  end

  % 9. Compute NMSD and plot first 5000 iters
  nmsd = 10*log10( sum((H_store - b(:)).^2,1) / (norm(b)^2+eps) );
  plot(1:5000, nmsd(1:5000), 'LineWidth',1.2, 'Color',colors(idxS,:));
end

xlabel('Iteration');
ylabel('NMSD (dB)');
title('SSM-INSAF Convergence vs. SNR (M=64, P=2, \beta=0.9)');
legend(arrayfun(@(s)sprintf('SNR=%d dB',s),SNRdB_list,'Uni',0), ...
       'Location','northeast');
grid on; ylim([-50,10]);
