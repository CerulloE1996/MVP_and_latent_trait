functions {
 
 
 
 

  real Phi_approx_2(real b, real boundary_for_rough_approx) {
    real a;
    if (abs(b) < boundary_for_rough_approx) { // i.e. NOT in the tails of N(0, 1)
      a = Phi( b);
    } else { 
      a = inv_logit(1.702 * b);
    }
      return a;
  } 
 
 
  
 
  
  
  
  
  
    real phi_approx(real b) {
    real a;
    a = inv_logit(1.702 * b);
    return a;
  }
  vector phi_approx_vec(vector b) {
    int N = num_elements(b);
    vector[N] a;
    a = inv_logit(1.702 * b);
    return a;
  }
  real inv_phi_approx(real b) {
    real a;
    a = logit(b) / 1.702;
    return a;
  } 
  vector inv_phi_approx_vec(vector b) {
    int N = num_elements(b);
    vector[N] a;
    a = logit(b) / 1.702;
    return a;
  }
  
  matrix cov2cor(matrix V) {
    int p = rows(V);
    vector[p] Is = inv_sqrt(diagonal(V));
    return quad_form_diag(V, Is); 
  }
  
  vector lower_elements(matrix M, int tri_size) {
    int n = rows(M);
    int counter = 1;
    vector[tri_size] out;
    for (i in 2 : n) {
      for (j in 1 : (i - 1)) {
        out[counter] = M[i, j];
        counter += 1;
      }
    }
    return out;
  }
  
  


  
  real tanh_1(real x) { 
    return ( ( 2 / (1+ exp(-2*x)) ) - 1 ) ; 
    }
    
  vector tanh_1(vector x) { 
    return ( ( 2 / (1+ exp(-2*x)) ) - 1 ) ; 
    }
    
  
   vector lb_ub_lp (vector y, real lb, real ub) {
 
    int N = num_elements(y); 
    vector[N] tanh_y; 
   // tanh_y = tanh_1(y);
    tanh_y = tanh(y);
      target +=  - log(2)  +  log( (ub - lb) * (1 - square(tanh_y))) ;
    return lb +  (ub - lb) *  0.5 * (1 + tanh_y) ; 
    
  }
 
    real lb_ub_lp (real y, real lb, real ub) {
      
       real  tanh_y = tanh(y);
    //   real tanh_y = tanh_1(y);
       target +=  - log(2)  +  log( (ub - lb) * (1 - square(tanh_y))) ;
      return lb +  (ub - lb) *  0.5 * (1 + tanh_y) ;
    
  }
  
  
  real inv_logit_1(real z) {
    return   (1/(1 +   exp(-z) )) ;
  }
  
   vector inv_logit_1(vector z) {
        return   (1/(1 +   exp(-z) )) ;
  }
  
   real  logit_1(real z) {
    return  log(z / ( 1 - z) ) ; 
  }
  
   vector logit_1(vector z) {
     return   log(z ./ ( 1 - z) ) ; 
  }
  
  
 real Phi_approx_1(real z) {
    return   (1/(1 + exp(-(0.07056*z.*z.*z + 1.5976*z )))) ;
  }
  
 vector Phi_approx_1(vector z) {
      return   (1/(1 + exp(-(0.07056*z.*z.*z + 1.5976*z )))) ;
 }
 
 
 real Phi_using_erfc(real z, real minus_sqrt_2_recip) {
  // real minus_sqrt_2_recip =  - 1 / sqrt(2);
  return 0.5 *  erfc( minus_sqrt_2_recip * z ) ; 
  }
  
 vector Phi_using_erfc(vector z, real minus_sqrt_2_recip) {
  // real minus_sqrt_2_recip =  - 1 / sqrt(2);
  return 0.5 *  erfc( minus_sqrt_2_recip * z ) ; 
  }
  
  
  real sinh_1(real x) { 
       real exp_x_i = exp(x); 
      //  real exp_2x_i = exp_x_i*exp_x_i;
       
   return  ((exp_x_i*exp_x_i) - 1)/(2*exp_x_i) ; 
   
    }
    
   real arcsinh_1(real x) { 
       
      return    log(x + sqrt(1 + (x*x)))  ; 
   
    }
    
    
 
    
  real  inv_Phi_approx_1(real p) { 
    return 5.494 * sinh_1(arcsinh_1(-0.3418*log(1/p-1))/3)  ; 
    }
    
  
 
  
}

  
 
  
 


data {
  
  int<lower=1> N;
  int<lower=2> n_tests;
  int n_ordinal_tests;
  int n_binary_tests;
  int<lower=2> n_class;// this version only works for 2 classes.
  array[n_class] int vector_classes;
  int<lower=1> n_pops;
  array[N] int pop;
  array[n_pops] real<lower=0> prior_p_alpha;
  array[n_pops] real<lower=0> prior_p_beta;
  array[n_tests, N] int<lower=0> y;  //////// data
  int<lower=0, upper=1> homog_corr;
  int<lower=0, upper=1> cons_tests_corr;
  matrix[n_class, n_tests]  prior_b_shape;
  matrix[n_class, n_tests]  prior_b_scale;
  int prior_only;
 // int<lower=1> K;
 // matrix[K, N] x; /////// covariates
  matrix[n_class, n_tests]  prior_a_mean;
  matrix<lower=0>[n_class, n_tests]  prior_a_sd;
  real boundary_for_rough_approx; 
  int LT_prior_on_mean_accuracy ; 
  
}


parameters {
       matrix[n_class, n_tests] LT_a;
       matrix<lower=0>[n_class, n_tests] LT_b;
       vector[n_pops]  p_raw;
       matrix[n_tests, N] u_raw; // nuisance that absorbs inequality constraints
}


transformed parameters {
  
  matrix[n_class, n_tests] theta;
  vector[N] log_lik;
  vector<lower=0, upper=1>[n_pops]  p;
  array[n_class] matrix[n_tests, n_tests] Omega;
  array[n_class] matrix[n_tests, n_tests] L_Omega;
   
   real minus_sqrt_2_recip = - (1.0 / sqrt(2.0)) ; 
   p =  lb_ub_lp(p_raw, 0.0, 1.0);
   
      
      for (t in 1 : n_tests) {
             for (c in 1 : n_class) {
                 theta[c, t] = LT_a[c, t] / sqrt(1.0 + square( LT_b[c, t]) ) ; 
             }
      }
                                    
    for (c in 1:n_class) {
          if (homog_corr == 0 && cons_tests_corr == 0) 
              Omega[c, ] = diag_matrix(rep_vector(1,n_tests)) + to_vector(LT_b[c, ]) * to_vector(LT_b[c, ])';
          if (homog_corr == 1 && cons_tests_corr == 0) 
              Omega[c, ] = diag_matrix(rep_vector(1,n_tests)) + to_vector(LT_b[1, ]) * to_vector(LT_b[1, ])';
          if (homog_corr == 0 && cons_tests_corr == 1) 
              Omega[c, ] = diag_matrix(rep_vector(1,n_tests))  + rep_vector(LT_b[c, 1], n_tests) * rep_vector(LT_b[c, 1], n_tests)';
          if (homog_corr == 1 && cons_tests_corr == 1) 
              Omega[c, ] = diag_matrix(rep_vector(1,n_tests))  + rep_vector(LT_b[1, 1], n_tests) * rep_vector(LT_b[1, 1], n_tests)';
              
        L_Omega[c, ] = cholesky_decompose(Omega[c, ]);
      }

  {
                      
                          vector[n_class] lp;
                          matrix[n_tests, n_class] z;
                          matrix[n_class, n_tests] y1;
                          matrix[n_class, n_binary_tests] bound_bin;
                          matrix[n_class, n_tests] Xbeta_n;
                          
            // Parameters for likelihood function. Based on code upoaded by Ben Goodrich which uses the 
            // GHK algorithm for generating TruncMVN. See: https://github.com/stan-dev/example-models/blob/master/misc/multivariate-probit/probit-multi-good.stan#L11
                if (prior_only == 0) {
                for (n in 1 : N) {
                  
                   vector[n_class] prev = rep_vector(0, n_class);
                          
                          for (t in 1 : n_tests) {
                            for (c in 1 : n_class) {

                                  real u = lb_ub_lp((u_raw[t,n]), 0.0, 1.0) ;
                                  //     real u = u_constrained[t,n]; 
                                  real stuff = - ( LT_a[c, t] + prev[c])  /  L_Omega[c, t, t] ;
                              
                                   if   ( abs(stuff) < boundary_for_rough_approx)   { 
                                     
                                            bound_bin[c, t] =    Phi(  stuff )   ; //   Phi_approx(  stuff   );
                                            if (y[t, n] == 0) {
                                                z[t, c] = inv_Phi( bound_bin[c, t] *  u );
                                               y1[c, t] = log(bound_bin[c, t]); // Jacobian adjustment
                                            } else if (y[t, n] == 1) {
                                                    z[t, c] = inv_Phi( bound_bin[c, t]      + (1.0 - bound_bin[c, t])  * u );
                                                   y1[c, t] = log1m(bound_bin[c, t]); // Jacobian adjustment
                                            } else {  // y == missing
                                              z[t, c] = L_Omega[c, t, t] * inv_Phi(u);
                                              y1[c, t] = 0.0;
                                            }
                                            
                                     }  else  { 
                                       
                                             bound_bin[c, t] = inv_logit(  1.702 * stuff   );
                                               
                                            if (y[t, n] == 0) {
                                              z[t, c] = logit( bound_bin[c, t]   * u )  / 1.702;
                                               y1[c, t] = log(bound_bin[c, t]); // Jacobian adjustment
                                            } else if (y[t, n] == 1) {
                                                z[t, c] = logit( bound_bin[c, t]        + (1.0 - bound_bin[c, t])  *  u )  / 1.702;
                                                y1[c, t] = log1m(bound_bin[c, t]); // Jacobian adjustment
                                            } else {  // y == missing
                                                z[t, c] = L_Omega[c, t, t] * logit(u) / 1.702 ;
                                                y1[c, t] = 0.0;
                                            }
                                       } 
                                
                              if (t < n_tests) {
                                prev[c] = L_Omega[c, t + 1, 1 : t]  * head( z[:,c], t);
                              }
                            }
                          }
                          // Jacobian adjustments imply z is truncated standard normal
                          // thus utility --- mu + L_Omega * z --- is truncated multivariate normal
                          for (c in 1 : n_class) {
                              lp[c] = sum(y1[c,  : ]) + bernoulli_lpmf(c - 1 | p[pop[n]]);
                          }
                          log_lik[n] = log_sum_exp(lp);
                          
                }
              }
  }
  
}


model {
       
        for (c in 1 : n_class) {
            for (t in 1 : n_tests) {
                if (LT_prior_on_mean_accuracy == 1) { 
                          theta[c, t] ~ normal(prior_a_mean[c, t], prior_a_sd[c, t]);
                          target += -0.5 * log(1 + square(LT_b[c, t]) ); 
                }  else   {  // doesnt need Jacobian as setting prior directly on RAW model parameter (NOT on transformed parameter)
                          LT_a[c, t]  ~ normal(prior_a_mean[c, t], prior_a_sd[c, t]);
                }
                LT_b[c, t] ~ weibull(prior_b_shape[c, t], prior_b_scale[c, t]);
            }
        }
        
        
        if (prior_only == 0) {
              for (g in 1 : n_pops) {
                p[g] ~ beta(prior_p_alpha[g], prior_p_beta[g]);
              }
              
                for (n in 1 : N) {
                  target += log_lik[n];
                }
        }
  
}
 
generated quantities {
  
    vector[n_tests] Se_median;
    vector[n_tests] Sp_median;
    vector[n_tests] Fp_median;
    vector[n_tests] Se_mean;
    vector[n_tests] Sp_mean;
    vector[n_tests] Fp_mean;

   for (c in 1:n_class) {
      for (t in 1:n_tests) { // for binary tests
      
              Se_median[t]  =        Phi_approx_2(   LT_a[2, t]  , boundary_for_rough_approx );
              Sp_median[t]  =    1 - Phi_approx_2(   LT_a[1, t]  , boundary_for_rough_approx );
              Fp_median[t]  =    1 - Sp_median[t];
              
              Se_mean[t]  =        Phi_approx_2(   LT_a[2, t]  / sqrt(1 + square(LT_b[2, t])) , boundary_for_rough_approx );
              Sp_mean[t]  =    1 - Phi_approx_2(   LT_a[1, t]  / sqrt(1 + square(LT_b[1, t])) , boundary_for_rough_approx );
              Fp_mean[t]  =    1 - Sp_mean[t];
              
    }
    
}

}
 
