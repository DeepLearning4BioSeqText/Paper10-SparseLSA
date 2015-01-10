function [U,A, density]=sparse_LSA(X, ReduceDim, lambda, init, g_opts)
    
    if isfield(init, 'n_alter')
        n_alter=init.n_alter;
    else
        n_alter=100;
    end
    
    if isfield(init, 'U_tol')
        U_tol=init.U_tol;
    else
        U_tol=1e-2;        
    end

    if isfield(init, 'A_tol')
        A_tol=init.A_tol;
    else
        A_tol=1e-2;        
    end    
    
    if isfield(init, 'A')
        A_old=init.A;
    else 
        A_old=zeros(ReduceDim, size(X,2));
    end    
    
    if isfield(init, 'U')
        U_old=init.U;
    else
        U_old=eye(size(data,1), ReduceDim);
    end    
    
    if isfield(init, 'group_tag') %whether we have group structure
        group_tag=init.group_tag;
    else
        group_tag=false;
    end
    
    if isfield(init, 'nn_tag') % whether it is nonnegative
        nn_tag=init.nn_tag;
    else
        nn_tag=false;
    end  
   
    if group_tag
        ind=init.ind;
        gWeight=init.gWeight;        
    end
    density=1;
    
    for iter=1:n_alter
        
        %learning sparse A
        if (lambda>0)
            if (group_tag)
                A=A_learning_group(lambda, X, U_old, ind, gWeight, g_opts);
            elseif (nn_tag)
                A=A_learning_nn(lambda, X, U_old);
            else
                A=A_learning(lambda, X, U_old);
            end
            density=sum(abs(A(:))>eps)/length(A(:));            
        else
            A=U_old'*X;        
        end

        %learning U        
        [P,D,Q]=lansvd(X*A', ReduceDim, 'L');  
        %% change to [P,D,Q]=svds(X*A', ReduceDim, 'L') if PROPACK cannot be used;
        U=P*Q';
        
        U_diff=norm(U-U_old, 'fro');
        A_diff=norm(A-A_old, 'fro');
        
        obj=norm(X-U*A, 'fro')^2/2+lambda*sum(abs(A(:))); 
        % if practice, one can avoid computation of objective, print obj just for tracking the algorithm. 
        fprintf('Alter: %d, obj: %.3f, density_A: %.5f, U_diff: %.5f, A_diff: %.5f\n', iter, obj,  density, U_diff, A_diff);       
            
        
        if (U_diff<U_tol && A_diff<A_tol)
            break;
        end
        
        U_old=U;
        A_old=A;
    end
end