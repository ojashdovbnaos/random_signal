%% debug_sm_ssm_full.m
% SM / SSM / Full-INSAF comparison with update rate statistics and debugging info
clear; clc; close all; rng(1);

%% Parameter Settings
fs = 8000; T = 5; L = fs*T; M = 512; Nsub = 16; Lp = 256;
mu0 = 0.3; delta = 1e-6; t_bound = 2; P = 2; beta = 0.9; SNRdB = 20;

%% AR(1) Input & Echo Path
w_noise = randn(L,1);
u = filter(1,[1 -0.9], w_noise);
w_true = fir1(M-1, 0.1)';
clean = filter(w_true, 1, u);
noiseP = var(clean)/10^(SNRdB/10);
d = clean + sqrt(noiseP)*randn(L,1);

%% Cosine Modulated Filter Bank
proto = fir1(Lp-1,1/Nsub,hamming(Lp))*(2/sqrt(Nsub));
h = zeros(Nsub, Lp);
for i = 0:Nsub-1
  for n = 0:Lp-1
    h(i+1,n+1) = 2 * proto(n+1) * cos((2*i+1)*pi/(4*Nsub)*(2*n-(Lp-1)) + (-1)^i*pi/4);
  end
end

%% Subband Decomposition (no downsampling)
u_sb = zeros(L, Nsub); d_sb = zeros(L, Nsub); nsb = zeros(L, Nsub);
noise = sqrt(noiseP)*randn(L,1);
for i = 1:Nsub
  u_sb(:,i) = filter(h(i,:),1,u);
  d_sb(:,i) = filter(h(i,:),1,d);
  nsb(:,i)  = filter(h(i,:),1,noise);
end
sigma_eta = var(nsb,0,1)';
gamma_i = t_bound * sqrt(sigma_eta);

%% Initialization
W0 = zeros(M, P); step = Nsub; total_it = floor((L-M)/step);
nmsd_sm = zeros(1,total_it);
nmsd_ssm = zeros(1,total_it);
nmsd_full = zeros(1,total_it);
upd_rate = struct('sm', zeros(1,Nsub), 'ssm', zeros(1,Nsub), 'full', ones(1,Nsub));

%% Main Loop
types = {'sm', 'ssm', 'full'};
for mode = types
  tag = mode{1};
  W_past = W0; sigma_eps = zeros(Nsub,1); upd_cnt = zeros(Nsub,1);
  for idx = 1:total_it
    k = M + (idx-1)*step;
    wbar = mean(W_past,2); dW = zeros(M,1);
    for i = 1:Nsub
      uvec = u_sb(k:-1:k-M+1,i);
      e_i = d_sb(k,i) - wbar' * uvec;

      do_update = true; mu_i = 1;
      if tag == "sm"
          if abs(e_i) <= gamma_i(i)
              do_update = false;
          else
              mu_i = 1 - gamma_i(i)/max(abs(e_i),1e-12);
          end
      elseif tag == "ssm"
          sigma_eps(i) = beta*sigma_eps(i) + (1-beta)*abs(e_i);
          if min(abs(e_i), sigma_eps(i)) <= gamma_i(i)
              do_update = false;
          else
              mu_i = 1 - gamma_i(i)/max(sigma_eps(i),1e-12);
          end
      end

      if idx == 1 && i <= 3 && strcmp(tag,'sm')
          fprintf('[SM] Subband %d: |e_i| = %.3e, gamma = %.3e\n', i, abs(e_i), gamma_i(i));
      end

      if do_update
        dW = dW + (mu0 * mu_i * e_i * uvec)/(uvec'*uvec + delta);
        upd_cnt(i) = upd_cnt(i) + 1;
      end
    end
    w_new = wbar + dW;
    W_past = [w_new, W_past(:,1:end-1)];
    nmsd = 10*log10( norm(w_true - w_new)^2 / (norm(w_true)^2 + eps) );
    switch tag
      case 'sm', nmsd_sm(idx) = nmsd;
      case 'ssm', nmsd_ssm(idx) = nmsd;
      case 'full', nmsd_full(idx) = nmsd;
    end
  end
  if ~strcmp(tag,'full')
      upd_rate.(tag) = upd_cnt / total_it;
      fprintf("[%s] Avg update rate = %.2f%%\n", upper(tag), mean(upd_cnt)/total_it*100);
  end
end

%% Plot (iteration as x-axis)
iters = 1:total_it;
figure;
plot(iters, nmsd_sm,'-b','LineWidth',1.5); hold on;
plot(iters, nmsd_ssm,'-g','LineWidth',1.5);
plot(iters, nmsd_full,'-r','LineWidth',1.5);
legend('SM-INSAF','SSM-INSAF','Full-INSAF');
xlabel('Iteration'); ylabel('NMSD (dB)');
title('Pseudo-Downsampled INSAF Variants (vs. Iteration)');
grid on; ylim([-50 5]);
