clear;
addpath(genpath('./PROPACK'));

% load data
option.filename='20ng';

doc_filename=['./', option.filename, '_data.mat'];
load(doc_filename);

[n,m]=size(data);

% latent dimension
ReduceDim=10;

init.n_alter=100;  % number of outer iterations
init.U_tol=1e-1;  % tolerance for U matrix (for convergence)
init.A_tol=1e-1;  % tolerance for A matrix (for convergence)
SVD_tag=false;    % initilization from SVD 

lambda=0.1;
init.nn_tag=false; %non-negative sparse LSA?
init.group_tag=false; %group sparse LSA.

if (SVD_tag)
    % initilization from SVD results
    [init.U, init.S, init.V]=lansvd(data, ReduceDim, 'L', OPTIONS);        
    init.S=[];
    init.V=[];    
else    
    % initlization U as the identity matrix
    init.U=eye(size(data,1), ReduceDim);    
end

tic;
    [U, dense_A, density]=sparse_LSA(data, ReduceDim, lambda, init);
    %init.nn_tag=true;  % nonnegative sparse LSA
    %[U, dense_A, density]=sparse_LSA(data, ReduceDim, lambda, init);
t=toc

ind=find(dense_A~=0);
[I, J]=ind2sub([ReduceDim,m], ind);
A=sparse(I, J, dense_A(ind), ReduceDim, m);
clear dense_A
     
