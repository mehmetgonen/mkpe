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

%initialize the kernels and data matrices
K_c = ??; %should be an N_x x N_z matrix containing cross-domain interactions between samples of domains X and Z
K_x = ??; %should be an N_x x N_x matrix containing within-domain similarities between samples of domain X
K_z = ??; %should be an N_z x N_z matrix containing within-domain similarities between samples of domain Z
X_train = ??; %should be an N_x x D_x matrix, which will be used for projecting training samples of domain X into subspace
Z_train = ??; %should be an N_z x D_z matrix, which will be used for projecting training samples of domain Z into subspace

%perform training
state = mkpe_projection_train(K_c, K_x, K_z, X_train, Z_train, parameters);

%initialize the data matrices for testing
X_test = ??; %should be an N_x x D_x matrix, which will be used for projecting samples of domain X into subspace
Z_test = ??; %should be an N_z x D_z matrix, which will be used for projecting samples of domain Z into subspace

%perform prediction
prediction = mkpe_projection_test(X_test, Z_test, state);

%display the embeddings for each domain
display(prediction.E_x);
display(prediction.E_z);
