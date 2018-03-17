%initalize the parameters of the algorithm
parameters = struct();

%set the convergence threshold parameter
parameters.epsilon = -Inf;

%set the maximum number of iterations
parameters.iteration = 100;

%set the regularization parameter for cross-domain interactions
parameters.lambda_c = 1.0;

%set the regularization parameters for within-domain similarities
parameters.lambda_x = 0.1;
parameters.lambda_z = 0.1;

%determine whether you want to learn the kernel width used in the subspace
parameters.learn_sigma_e = 1;

%set the subspace dimensionality
parameters.R = 2;

%set the seed for random number generator used to initalize random variables
parameters.seed = 1606;

%set the kernel width used in the subspace
parameters.sigma_e = sqrt(parameters.R);

%initialize the kernels
K_c = rand(50, 40); %should be an N_x x N_z matrix containing cross-domain interactions between samples of domains X and Z
K_x = rand(50, 50); %should be an N_x x N_x matrix containing within-domain similarities between samples of domain X
K_z = rand(40, 40); %should be an N_z x N_z matrix containing within-domain similarities between samples of domain Z

%perform training
state = mkpe_embedding_train(K_c, K_x, K_z, parameters);

%display the embeddings for each domain
display(state.E_x);
display(state.E_z);
