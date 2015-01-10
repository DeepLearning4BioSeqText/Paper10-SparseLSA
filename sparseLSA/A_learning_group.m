function B=A_learning_group(lambda, Y, X, B, ind, gWeight, g_opts)    
        
    n_g=length(gWeight);
    D=size(X,2);
    
    if exist('g_opts', 'var') && isfield(g_opts, 'max_iter')
        max_iter=g_opts.max_iter;
    else 
        max_iter=50;
    end
    
    if exist('g_opts', 'var') && isfield(g_opts, 'tol')
        tol=g_opts.tol;
    else 
        tol=1e-3;
    end
    
    if exist('g_opts', 'var') && isfield(g_opts, 'verbose')
        verbose=g_opts.verbose;
    else 
        verbose=false;
    end
    
    c=sum(X.^2,1);
    gWeight=gWeight*lambda;
    
    
    obj_old=sum(sum((Y-X*B).^2))/2+cal2norm(B,ind,gWeight); 
    density_old=sum(B(:)~=0)/length(B(:));
    if (verbose)
       fprintf('Iter: %d, Obj: %g, Den: %g\n', 0, obj_old, density_old);    
    end
    
    for iter=1:max_iter
       
       for s=1:D
           for g=1:n_g
               g_ind=ind(g)+1:ind(g+1);               
               alpha=X(:,s)'*(Y(:,g_ind)-X*B(:,g_ind)+X(:,s)*B(s,g_ind));
               alpha_norm=sqrt(sum(alpha.^2));
               if (alpha_norm>gWeight(g))
                   B(s,g_ind)=alpha*(alpha_norm-gWeight(g))/(c(s)*alpha_norm);
               else
                   B(s,g_ind)=0;
               end
           end
       end
       
       obj=sum(sum((Y-X*B).^2))/2+cal2norm(B,ind,gWeight); 
       density=sum(B(:)~=0)/length(B(:));
       if (verbose)
            fprintf('Iter: %d, Obj: %g, Den: %g\n', iter, obj, density);    
       end
       
        if (iter>1 && abs(obj-obj_old)<tol)
            break;
        else 
            obj_old=obj;
        end
    end    
end
