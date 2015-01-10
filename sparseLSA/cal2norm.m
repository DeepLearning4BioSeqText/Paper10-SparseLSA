function s=cal2norm(B, ind, gWeight)
         n_g=length(gWeight);
         s=0;
         for g=1:n_g
             g_ind=ind(g)+1:ind(g+1);
             gnorm=sqrt(sum(B(:,g_ind).^2,2));
             s=s+gWeight(g)*sum(gnorm);
         end
end