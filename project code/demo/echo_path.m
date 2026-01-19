clear; clc; close all;

M = 512;  

w_a = zeros(M,1);
tap_locs = [1, 20, 35, 50, 80, 100, 130];  
w_a(tap_locs) = [0.7, -0.3, 0.2, -0.15, 0.1, -0.08, 0.05];


w_b = 0.2 * randn(M,1);
w_b(1:10) = w_b(1:10) + linspace(0.4, 0, 10)';  % 注意转置


figure('Position',[200 200 420 520]);

subplot(2,1,1);
stem(1:M, w_a, 'b','Marker','none');
ylabel('Magnitude');
title('Sparse System (a)');
xlim([0 M]);
ylim([-0.2 0.8]);

subplot(2,1,2);
stem(1:M, w_b, 'b','Marker','none');
xlabel('Tap index'); ylabel('Magnitude');
title('Dense System (b)');
xlim([0 M]);
ylim([-0.2 0.8]);


sgtitle('Impulse Responses of Echo');

