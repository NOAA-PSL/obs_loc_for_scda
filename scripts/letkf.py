import numpy as np

def calculate_normalized_distance(pos_state, pos_ob, loc_rad_state, loc_rad_ob):
  """Computes distance between state variable and observations normalized by localization radii
  
  Inputs:
  pos_state -- position of the state variable
  pos_ob -- position of the observations
  loc_rad_state -- localization radius associated with the state variable
  loc_rad_ob -- localization radius associated with the observations

  Outputs:
  norm_dist -- normalized distance
  """
  norm_dist = np.abs(np.divide(pos_state,loc_rad_state) - np.divide(pos_ob,loc_rad_ob))
  return norm_dist


def select_local_obs(norm_dist, *args):
  """Selects observations within a certain radius of a given grid point
  
  Inputs:
  norm_dist -- normalized distance between state variables and observations
  *args -- full data sets from which to extract local observations

  Outputs:
  args -- local observations
  """
  select_these = norm_dist <= 1
  for arg in args:
    arg = arg[select_these, ]
  return args

def gaspari_cohn(dist):
  """Computes Gaspari-Cohn 5th order localization function
  """
  gc = np.empty_like(dist)
  # Three cases
  less1 = dist < 1
  less2 = np.logical_and(dist >= 1, dist < 2)
  geq2 = dist > 2
  # Compute GC function
  gc[less1] = -0.25*dist[less1]**5 + .5*dist[less1]**4 + 0.625*dist[less1]**3 - 5/3*dist[less1]**2 + 1;
  gc[less2] = 1/12*dist[less2]**5 - .5*dist[less2]**4 + 0.625*dist[less2]**3 + 5/3*dist[less2]**2 - 5*dist[less2] + 4 - 2/3*dist[less2]**-1 ;
  gc[geq2] = 0
  return gc

def separate_mean_pert(ens):
  """Separate ensemble mean and ensemble perturbations

  Input:
  ens -- an ensemble

  Outputs:
  avg -- ensemble mean
  pert -- ensemble perturbations
  """
  ens = np.atleast_2d(ens)
  ens_size = ens.shape[1]
  avg = np.mean(ens, 1)
  pert = ens - np.tile(avg, (ens_size, 1)).transpose()

  return avg, pert 

def letkf_inner_loop(which_state, x_avg, x_ens, y_avg, y_ens, R_inv, y_ob, pos_state, pos_ob, loc_rad_state, loc_rad_ob, inflate, use_loc=True):
  """Compute the LETKF update for a single grid point

  Inputs:
  which_state -- the index(ices) of the state variable(s) to be updated
  x_avg -- background ensemble mean
  x_ens -- background perturbation ensemble
  y_avg -- background observation ensemble mean
  y_ens -- background observation perturbation ensemble
  R_inv -- inverse observation error variances (stored as 1d object)
  y_ob -- observations
  pos_state -- positions of state variables
  pos_ob -- positions of the observations
  loc_rad_state -- localization radius associated with state variable
  loc_rad_ob -- localization radius associated with observations
  inflate -- multiplicative inflation factor

  Output:
  analysis_ens -- analysis ensemble at a single grid point
  """
  # Get ensemble size
  ens_size = x_ens.shape[1]

  # Select local states
  x_avg = x_avg[which_state]
  x_ens = x_ens[which_state, :]
  pos_state = pos_state[which_state]
  loc_rad_state = loc_rad_state[which_state]

  # Select local observation(s)
  norm_dist = calculate_normalized_distance(pos_state, pos_ob, loc_rad_state, loc_rad_ob)
  select_these = norm_dist < 1  

  if np.any(select_these): # Update with local observations
    # Select local observations
    #y_avg = y_avg[[select_these,]]
    #y_ens = y_ens[[select_these,]]
    #y_ob = y_ob[[select_these,]]
    #R_inv = R_inv[[select_these,]]
    #loc_rad_ob = loc_rad_ob[[select_these,]]
    #norm_dist = norm_dist[[select_these]]
    
    # Compute ob space localization
    if use_loc:
      loc_inv = gaspari_cohn(2*norm_dist)    
      R_inv = R_inv * loc_inv

    # Compute C = y_ens^T * R_inv
    C = np.transpose(y_ens) @ np.atleast_2d(np.diag(R_inv))

    # Compute Pa = [(ens_size/rho)I + Cy_ens]^-1
    CY = C @ y_ens
    CY = 0.5 * (CY + np.transpose(CY))
    work = ((ens_size-1)/inflate) * np.eye(ens_size) + CY
    evals, evecs = np.linalg.eigh(work)
    evals_inv = np.reciprocal(evals)
    Pa = evecs @ np.diag(evals_inv) @ np.transpose(evecs)

    # Compute Wa = [(ens_size - 1)Pa]^1/2
    Wa = evecs @ np.diag(np.sqrt((ens_size - 1) * evals_inv)) @ np.transpose(evecs)

    # Compute wa_bar = Pa @ C @ (y_ob - y_avg) and add to each column of Wa
    wa_bar = Pa @ C @ (y_ob - y_avg)
    Wa = Wa + np.tile(wa_bar, (1, ens_size))

    # Create analysis ensemble members
    analysis_ens = np.tile(x_avg, (1, ens_size)) + x_ens @ Wa
  
  else: # If no local observations do not update
    analysis_ens = np.tile(x_avg, (1, ens_size)) + x_ens  

  return analysis_ens

def letkf_one_domain(x_ens, HofX, R_inv, y_ob, pos_state, pos_ob, loc_rad_state, loc_rad_ob, inflate, use_loc=True):
  """Compute the LETKF update with a given observation operator HofX

  Inputs:
  x_ens -- background ensemble
  HofX -- linear observation operator
  R_inv -- inverse observation error variances (stored as 1d object)
  y_ob -- observations
  pos_state -- positions of state variables
  pos_ob -- positions of the observations
  loc_rad_state -- localization radius associated with state variable
  loc_rad_ob -- localization radius associated with observations
  inflate -- multiplicative inflation factor

  Outputs:
  analysis_mean -- analysis mean
  analysis_cov -- analysis covariance matrix
  analysis_ens -- analysis ensemble
  """
  # Get domain size
  num_levels, ens_size = x_ens.shape
 
  # Form background observation perturbations
  y_ens = HofX @ x_ens
  y_avg, y_ens = separate_mean_pert(y_ens)
  
  # Form background ensemble perturbations
  x_avg, x_ens = separate_mean_pert(x_ens)

  # Loop over each vertical level
  analysis_ens = np.empty_like(x_ens)
  for ind in range(num_levels):
    analysis_ens[ind,] = letkf_inner_loop(ind, x_avg, x_ens, y_avg, y_ens, R_inv, y_ob, pos_state, pos_ob, loc_rad_state, loc_rad_ob, inflate, use_loc)

  # Compute analysis mean and covariance
  analysis_mean = np.mean(analysis_ens, 1)
  analysis_cov = np.cov(analysis_ens)

  return analysis_mean, analysis_cov 



def letkf(x_ens, y_ens, R_inv, y_ob, pos_state, pos_ob, loc_rad_state, loc_rad_ob, inflate, use_loc=True):
  """Compute the LETKF update with a given background observation ensemble

  Inputs:
  x_ens -- background ensemble
  y_ens -- background observation ensemble
  R_inv -- inverse observation error variances (stored as 1d object)
  y_ob -- observations
  pos_state -- positions of state variables
  pos_ob -- positions of the observations
  loc_rad_state -- localization radius associated with state variable
  loc_rad_ob -- localization radius associated with observations
  inflate -- multiplicative inflation factor

  Outputs:
  analysis_mean -- analysis mean
  analysis_cov -- analysis covariance matrix
  analysis_ens -- analysis ensemble
  """
  # Get domain size
  num_levels, ens_size = x_ens.shape
 
  # Form background observation perturbations
  y_avg, y_ens = separate_mean_pert(y_ens)
  
  # Form background ensemble perturbations
  x_avg, x_ens = separate_mean_pert(x_ens)

  # Loop over each vertical level
  analysis_ens = np.empty_like(x_ens)
  for ind in range(num_levels):
    analysis_ens[ind,] = letkf_inner_loop(ind, x_avg, x_ens, y_avg, y_ens, R_inv, y_ob, pos_state, pos_ob, loc_rad_state, loc_rad_ob, inflate, use_loc)

  # Compute analysis mean and covariance
  analysis_mean = np.mean(analysis_ens, 1)
  analysis_cov = np.cov(analysis_ens)

  return analysis_mean, analysis_cov 
