function A=A_learning(lambda, X, U)    
    
    UX=U'*X;
    A=sign(UX).*max(abs(UX)-lambda,0);
    
end
