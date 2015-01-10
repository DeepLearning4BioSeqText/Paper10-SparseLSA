function A=A_learning_nn(lambda, X, U)    
    
    A=max(U'*X-lambda, 0);

end
