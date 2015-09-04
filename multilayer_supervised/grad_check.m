% runs training procedure for supervised multilayer network
% softmax output layer with cross entropy loss function

%% setup environment
% experiment information
% a struct containing network layer sizes etc
ei = [];

% add common directory to your path for
% minfunc and mnist data helpers
addpath ../common;
addpath(genpath('../common/minFunc_2012/minFunc'));

%% load mnist data
[data_train, labels_train, data_test, labels_test] = load_preprocess_mnist();

NTRAIN = 100;
NGRAD = 100;
data_train = data_train(:,1:NTRAIN);
labels_train = labels_train(1:NTRAIN);

%% populate ei with the network architecture to train
% ei is a structure you can use to store hyperparameters of the network
% the architecture specified below should produce  100% training accuracy
% You should be able to try different network architectures by changing ei
% only (no changes to the objective function code)

% dimension of input features
ei.input_dim = 784;
% number of output classes
ei.output_dim = 10;
% sizes of all hidden layers and the output layer
ei.layer_sizes = [256, ei.output_dim];
% scaling parameter for l2 weight regularization penalty
ei.lambda = 0;
% which type of activation function to use in hidden layers
% feel free to implement support for only the logistic sigmoid function
ei.activation_fun = 'logistic';

%% setup random initial weights
stack = initialize_weights(ei);
params = stack2params(stack);

%% check gradients
EPSILON = 1e-4;
[~,grad] = supervised_dnn_cost(params, ei, data_train, labels_train, ...
                               false);

errors = 0;
avg_err = 0;
for k = 1:NGRAD
    i = randi(numel(params));
    oldtheta = params(i);
    params(i) = oldtheta + EPSILON;
    cost1 = supervised_dnn_cost(params, ei, data_train, ...
                                labels_train, false);

    params(i) = oldtheta - EPSILON;
    cost2 = supervised_dnn_cost(params, ei, data_train, ...
                                labels_train, false);

    approx_grad = (cost1-cost2)/(2*EPSILON);
    if abs(grad(i) - approx_grad) > 2*EPSILON
        fprintf(['grad(%d) did not match numerical calculation: %f ' ...
                 'instead of %f\n'], i, grad(i), approx_grad);
        errors = errors + 1;
        avg_err = avg_err + abs(grad(i) - approx_grad);
    else
        fprintf('Gradient check passed for grad(%d): %f ~= %f\n', i, ...
                grad(i), approx_grad);
    end
    params(i) = oldtheta;
end
if errors > 0
    avg_err = avg_err/errors;
end
fprintf('Gradient error rate: %f, Average Error: %f\n', errors/NGRAD, ...
        avg_err);