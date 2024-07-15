functions {

  matrix corr_to_chol(real x, int J) {
    matrix[J, J] cor = add_diag(rep_matrix(x, J, J), 1 - x);
    return cholesky_decompose(cor);
  }
  
  // Induced dirichlet from Betancourt, 2019. 
  // See https://betanalpha.github.io/assets/case_studies/ordinal_regression.html#3_cut_to_the_chase
  real induced_dirichlet_lpdf(vector c, vector alpha, real phi, real std) {
    int K = num_elements(c) + 1;
    vector[K - 1] anchoredcutoffs = (c - phi) / std;
    
    vector[K] sigma;
    vector[K] p;
    matrix[K, K] J = rep_matrix(0, K, K);
    
    sigma[1 : K - 1] = phi_approx_vec(anchoredcutoffs);
    sigma[K] = 1;
    
    p[1] = sigma[1];
    for (k in 2 : (K - 1)) {
      p[k] = sigma[k] - sigma[k - 1];
    }
    p[K] = 1 - sigma[K - 1];
    
    // Baseline column of Jacobian
    for (k in 1 : K) {
      J[k, 1] = 1;
    }
    
    // Diagonal entries of Jacobian
    for (k in 2 : K) {
      // rho is the PDF of the latent distribution
      real rho = 1.702 * sigma[k - 1] * (1 - sigma[k - 1]);
      J[k, k] = -rho;
      J[k - 1, k] = +rho;
    }
    
    return dirichlet_lpdf(p | alpha) + log_determinant(J);
  }
  
 
  vector corr_constrain_lp(vector x) {
    int N = num_elements(x);
    vector[N] tanh_x = tanh(x);
    
    target += sum(log1m(square(tanh_x)));
    
    return tanh_x;
    
  }
  
  
 
 
 
 

  real Phi_approx_2(real b, real boundary_for_rough_approx) {
    real a;
    if (abs(b) < boundary_for_rough_approx) { // i.e. NOT in the tails of N(0, 1)
      a = Phi( b);
    } else { 
      a = inv_logit(1.702 * b);
    }
      return a;
  } 
 
    
    
 
 
  //   vector lb_ub_lp (vector y, real lb, real ub) {
  //   target += log(ub - lb) + log_inv_logit(y) + log1m_inv_logit(y);
  //  
  //   return lb + (ub - lb) * inv_logit(y);
  // }
  // 
  //   real lb_ub_lp (real y, real lb, real ub) {
  //   target += log(ub - lb) + log_inv_logit(y) + log1m_inv_logit(y);
  //  
  //   return lb + (ub - lb) * inv_logit(y);
  // }
  // 
  //    vector lb_ub_lp (vector y, vector lb, vector ub) {
  //   target += log(ub - lb) + log_inv_logit(y) + log1m_inv_logit(y);
  //  
  //   return lb + (ub - lb) .* inv_logit(y);
  // }
  
  

 


 
 
 
  
  matrix cholesky_corr_constrain_outer_lp (vector col_one_raw, vector off_raw,
                                           real lb, real ub) {
    int K = num_elements(col_one_raw) + 1;
    vector[K - 1] z = lb_ub_lp(col_one_raw, lb, ub);
    matrix[K, K] L = diag_matrix(rep_vector(1, K));
    vector[K] D;
    D[1] = 1;
    L[2:K, 1] = z[1:K - 1];
    D[2] = 1 - L[2, 1]^2;
    int cnt = 1;
   
    for (i in 3:K) {
       D[i] = 1 - L[i, 1]^2; 
       L[i, 2:i - 1] = rep_row_vector(1 - L[i, 1]^2, i - 2);
       real l_ij_old = L[i, 2];
      for (j in 2:i - 1) {
      //  real l_ij_old = L[i, j];
        real b1 = dot_product(L[j, 1:(j - 1)], D[1:j - 1]' .* L[i, 1:(j - 1)]);
          
          // how to derive the bounds
          // we know that the correlation value C is bound by
          // b1 - Ljj * Lij_old <= C <= b1 + Ljj * Lij_old
          // Now we want our bounds to be enforced too so
          // max(lb, b1 - Ljj * Lij_old) <= C <= min(ub, b1 + Ljj * Lij_old)
          // We have the Lij_new = (C - b1) / Ljj
          // To get the bounds on Lij_new is
          // (bound - b1) / Ljj 
          
          real low = max({-sqrt(l_ij_old) * D[j], lb - b1});
          real up = min({sqrt(l_ij_old) * D[j], ub - b1});
          
          real x = lb_ub_lp(off_raw[cnt], low, up);
          L[i, j] = x / D[j]; 

          target += -0.5 * log(D[j]);
          
           l_ij_old *= 1 - (D[j] * L[i, j]^2) / l_ij_old;
          
         // real mul = 1 - (D[j] * L[i, j]^2) / l_ij_old;
         // L[i, (j + 1):i - 1] *= mul;
          //D[i] *= mul;
          cnt += 1;
        }
        D[i] = l_ij_old;
      }
        return diag_post_multiply(L, sqrt(D));
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
 
  //     int N = num_elements(z);
  //     vector[N] out;
  //     
  //     for (i in 1:N) { 
  //       real z_i = z[i];
  //       out[i] = (1/(1 + exp(-(0.07056*z_i*z_i*z_i + 1.5976*z_i )))) ;  
  //     }
  //     
  //     return out;
  // }
  
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
    
  


real qnorm_stan(real p) {
    real r;
    real val;
    real q = p - 0.5;
    
    if (abs(q) <= .425) {
      r = .180625 - q * q;
	    val = q * (((((((r * 2509.0809287301226727 +
                       33430.575583588128105) * r + 67265.770927008700853) * r +
                     45921.953931549871457) * r + 13731.693765509461125) * r +
                   1971.5909503065514427) * r + 133.14166789178437745) * r +
                 3.387132872796366608) / (((((((r * 5226.495278852854561 +
                     28729.085735721942674) * r + 39307.89580009271061) * r +
                   21213.794301586595867) * r + 5394.1960214247511077) * r +
                 687.1870074920579083) * r + 42.313330701600911252) * r + 1.0);
    } else { /* closer than 0.075 from {0,1} boundary */
	    if (q > 0) r = 1.0 - p;
	      else r = p;

	    r = sqrt(-log(r));
    
      if (r <= 5.) { /* <==> min(p,1-p) >= exp(-25) ~= 1.3888e-11 */
        r += -1.6;
        val = (((((((r * 0.00077454501427834140764 +
                       .0227238449892691845833) * r + .24178072517745061177) *
                     r + 1.27045825245236838258) * r +
                    3.64784832476320460504) * r + 5.7694972214606914055) * r + 4.6303378461565452959) * r +
                 1.42343711074968357734) / (((((((r *
                         0.00000000105075007164441684324 + 0.0005475938084995344946) *
                        r + .0151986665636164571966) * r +
                       .14810397642748007459) * r + .68976733498510000455) *
                     r + 1.6763848301838038494) * r +
                    2.05319162663775882187) * r + 1.);
        } else { /* very close to  0 or 1 */
            r += -5.;
            val = (((((((r * 0.000000201033439929228813265 +
                       0.0000271155556874348757815) * r +
                      .0012426609473880784386) * r + .026532189526576123093) *
                    r + .29656057182850489123) * r +
                   1.7848265399172913358) * r + 5.4637849111641143699) *
                 r + 6.6579046435011037772) / (((((((r *
                        0.00000000000000204426310338993978564 + 0.00000014215117583164458887)*
                        r + 0.000018463183175100546818) * r +
                       0.0007868691311456132591) * r + .0148753612908506148525)
                     * r + .13692988092273580531) * r +
                    .59983220655588793769) * r + 1.);
        }

	if(q < 0.0) val = -val;
  }
    return val;
}

 vector qnorm_stan_vec (vector p) {
   int N = num_elements(p);
   vector[N] out;
   
   for (n in 1:N) 
     out[n] = qnorm_stan(p[n]);
   
   return out;
 }
 



  
  
}

  
 
  
 


data {
  
  int<lower=1> N;
  int<lower=2> n_tests;
  int n_ordinal_tests;
  int n_binary_tests;
  int<lower=2> n_class;
  int mvr_cholesky;
  array[n_class] int vector_classes;
  int<lower=1> n_pops;
  array[N] int pop;
  array[n_pops] vector[n_class] prior_alpha_classes;
  array[n_pops] real<lower=0> prior_p_alpha;
  array[n_pops] real<lower=0> prior_p_beta;
  array[n_tests, N] int<lower=0> y;  //////// data
  int<lower=0, upper=1> homog_corr;
  int<lower=0, upper=1> cons_tests_corr;
  array[n_class, n_tests, n_tests] real prior_for_corr_a;
  array[n_class, n_tests, n_tests] real prior_for_corr_b;
  array[n_class] real prior_lkj;
  array[n_tests] int Thr;
  array[n_class, n_ordinal_tests, max(Thr) + 1] real prior_ind_dirichlet_array;
  array[n_class, n_ordinal_tests] real prior_anchor_point;
  int prior_only;
  int CI;
  int perfect_gs;
  int<lower=1> K;
  matrix[K, N] x; /////// covariates
  array[n_class, n_tests, K] real prior_a_mean;
  array[n_class, n_tests, K] real<lower=0> prior_a_sd;
  int induced_dir;
  int prior_cutpoint_sd;
  int rough_approx;
  int corr_force_positive;
  real corr_pareto_alpha;
  real corr_dist_mean; 
  real corr_dist_sd; 
  real boundary_for_rough_approx; 
  int<lower=0, upper=(n_tests * (n_tests - 1)) %/% 2> known_num;
  int  corr_prior_beta; 
  int  corr_prior_norm ; 
  
}

transformed data {
  
  int k_choose_2 = (n_tests * (n_tests - 1)) / 2;
  int km1_choose_2 = ((n_tests - 1) * (n_tests - 2)) %/% 2;
  
    real<lower=-1, upper=1> lb;
  real<lower=lb, upper=1> ub;
    
    ub = 1;
                  if (corr_force_positive == 1)  lb = 0;
                  else lb = -1;
  
}

parameters {
  
  array[n_class] matrix[n_tests, K] a;
  array[n_class] vector[n_tests - 1] col_one_raw;
  array[n_class] vector[km1_choose_2 - known_num] off_raw;
  vector[n_pops]  p_raw;
    matrix[n_tests, N] u_raw; // nuisance that absorbs inequality constraints
  //  matrix<lower=0,  upper=1>[n_tests, N] u_constrained; // nuisance that absorbs inequality constraints

}

transformed parameters {
  
  vector[N] log_lik;
  vector<lower=0, upper=1>[n_pops]  p;
  array[n_class] matrix[n_tests, n_tests] Omega;
  array[n_class] matrix[n_tests, n_tests] L_Omega;
    array[n_class] matrix[K, n_tests] a_transpose;
 // matrix[n_tests, N]   u; //  =   lb_ub_lp(to_vector(u_raw[,t]), 0, 1);
   
   real minus_sqrt_2_recip = - (1 / sqrt(2)) ; 
   p =  lb_ub_lp(p_raw, 0, 1);
   
   // for (t in 1 : n_tests) {
   //    u[t,] =   lb_ub_lp(to_vector(u_raw[t,]), 0, 1);
   // }
      
      for (t in 1 : n_tests) {
             for (c in 1 : n_class) {
               for (k in 1 : K) {
                  a_transpose[c, k, t] = a[c, t, k];
               }
             }
      }
        
      
    for (c in 1 : n_class) {
                              L_Omega[c, :  ] =   cholesky_corr_constrain_outer_lp( col_one_raw[c, :], off_raw[c, :], lb, ub);
                              Omega[c, :  ] = multiply_lower_tri_self_transpose(L_Omega[c, :]);
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
                                  Xbeta_n[c, t] =  a_transpose[c, 1,  t ] ; //  to_row_vector(x[:, n]) * (a_transpose[c, :,  t ]); // x[N,K] and a[C,T,K]
                            }
                       }
                          
                          for (t in 1 : n_tests) {
                            for (c in 1 : n_class) {

                                  real u = lb_ub_lp((u_raw[t,n]), 0, 1) ;
                                  //     real u = u_constrained[t,n]; 
                                  real stuff = - (Xbeta_n[c, t] + prev[c])  /  L_Omega[c, t, t] ;
                              
                                   if   ( abs(stuff) < boundary_for_rough_approx)   { 
                                     
                                            bound_bin[c, t] =    Phi(  stuff )   ; //   Phi_approx(  stuff   );
                                                 // inv_Phi_approx_1
                                                 // inv_Phi 
                                                 // erfc
                                                 // Phi_approx
                                                 // qnorm_stan
                                                 // c
                                            if (y[t, n] == 0) {
                                                z[t, c] = inv_Phi( bound_bin[c, t] *  u );
                                               y1[c, t] = log(bound_bin[c, t]); // Jacobian adjustment
                                            } else if (y[t, n] == 1) {
                                                    z[t, c] = inv_Phi( bound_bin[c, t]      + (1 - bound_bin[c, t])  * u );
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
                                                z[t, c] = logit( bound_bin[c, t]        + (1 - bound_bin[c, t])  *  u )  / 1.702;
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
          for (k in 1 : K) {
            for (t in 1 : n_tests) {
              a[c, t, k] ~ normal(prior_a_mean[c, t, k], prior_a_sd[c, t, k]);
            }
          }
        }
        
      
     
            // for (c in 1 : n_class) {
            // 
            //      if (corr_force_positive == 1)    target += sum(Omega_raw[c, : ]) ;   // both for beta  priors 
            //     // if (prior_only == 0)   target += sum(Omega_raw[c, : ]) ; // for likelihood  
            // 
            //                        
            //       //               
            //        int counter = 1;
            //         for (i in 2 : n_tests) {
            //          for (j in 1 : (i - 1)) {
            // 
            // 
            //                      if (corr_prior_beta == 1)    target +=  beta_lpdf( (Omega[c,i,j] + 1)/2 |  prior_for_corr_a[c, i, j]   , prior_for_corr_b[c, i, j]    );
            // 
            //                    counter = counter + 1;
            // 
            //                }
            //              }
            //              
            //                     vector[n_tests] jacobian_diag_elements;
            //                     for (i in 1 : n_tests)      jacobian_diag_elements[i] = (n_tests + 1 - i) * log(L_Omega[c, i, i]);
            //                      if (corr_prior_beta == 1)   target +=      +    (n_tests* log(2) + sum(jacobian_diag_elements) ); // for  beta prior  
            //       
            //       
            //         if (corr_prior_beta == 0)  target += lkj_corr_cholesky_lpdf(L_Omega[c, :] | 4) ; 
            //        
            //        
            //       }
                    // 
                       target += lkj_corr_cholesky_lpdf(L_Omega[1, :] | prior_lkj[1]) ;
                       target += lkj_corr_cholesky_lpdf(L_Omega[2, :] | prior_lkj[2]) ;
                    
                    
                   // for (c in 1 : n_class) {
                     //  
                     // for (i in 2 : n_tests) {
                     //      for (j in 1 : (i - 1)) {
                     // 
                     //        real alpha_for_beta_1 = prior_lkj[1] + (n_tests/2)  - 1 ; 
                     //        real alpha_for_beta_2 = prior_lkj[2] + (n_tests/2)  - 1 ; 
                     //        
                     //      //   target +=  beta_lpdf( (Omega[c,i,j] + 1)/2 |  prior_for_corr_a[c, i, j]   , prior_for_corr_b[c, i, j]    );
                     //         
                     //         
                     //         target += (( (beta_lpdf( (Omega[1,i,j] + 1)/2 |   alpha_for_beta_1   ,  alpha_for_beta_1    )) ))  ; 
                     //        // target +=                    log((1 / ( beta_cdf((1 + 1)/2  | alpha_for_beta_1, alpha_for_beta_1) -   beta_cdf((0 + 1)/2  | alpha_for_beta_1, alpha_for_beta_1) ))) ;
                     //        
                     //         target += (( (beta_lpdf( (Omega[2,i,j] + 1)/2 |   alpha_for_beta_2    , alpha_for_beta_2    )) ));// / 
                     //      //  target +=                   log((1 / ( beta_cdf((1 + 1)/2  | alpha_for_beta_2, alpha_for_beta_2) -   beta_cdf((0 + 1)/2  | alpha_for_beta_2, alpha_for_beta_2) ))) ;
                     //  
                     //      }
                     // }
                     //     
                     // 
                     // for (c in 1 : n_class) {
                     //             vector[n_tests] jacobian_diag_elements;
                     //             for (i in 1 : n_tests)      jacobian_diag_elements[i] = (n_tests + 1 - i) * log(L_Omega[c, i, i]);
                     //             target +=      +    (n_tests* log(2) + sum(jacobian_diag_elements) ); // for  beta prior
                     // }

    
 
        
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
  
    vector[n_binary_tests] Se_bin;
    vector[n_binary_tests] Sp_bin;
    vector[n_binary_tests] Fp_bin;
 

   for (c in 1:n_class) {
 

      for (t in 1:n_binary_tests) { // for binary tests

         if (n_class == 2) { // summary Se and Sp only calculated if n_class = 2 (i.e. the "standard" # of classes for DTA)
              Se_bin[t]  =        Phi_approx_2(   a[2, t, 1]  , boundary_for_rough_approx );
              Sp_bin[t]  =    1 - Phi_approx_2(   a[1, t, 1]  , boundary_for_rough_approx);
              Fp_bin[t]  =    1 - Sp_bin[t];
        }
        else {
          Se_bin[t] = 999;
          Sp_bin[t] = 999;
          Fp_bin[t] = 999;
        }
    }
    
 

}

}
 
