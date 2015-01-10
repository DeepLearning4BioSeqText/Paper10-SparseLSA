% warning('off','PROPACK:NotUsingMex')
randn('state',1);
M = 80; N = 90;
A = randn(M,50) * randn(50,N);
opt = []; opt.cgs = 0;
[U,S,V] = lansvd( @(x) A*x, @(y)A'*y, M, N, min([M,N]), 'L', opt );
[UU,SS,VV] = svd(A);  % compare with Matlab

s = diag(S); ss = diag(SS);
disp('PROPACK:   MATLAB:');
disp([ s(1:50), ss(1:50) ])
fprintf('discrepancy in singular values is %e\n',norm(s-ss));