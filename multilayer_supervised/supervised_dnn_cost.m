function [ cost, grad, pred_prob] = supervised_dnn_cost( theta, ei, data, labels, pred_only)
%SPNETCOSTSLAVE Slave cost function for simple phone net
%   Does all the work of cost / gradient computation
%   Returns cost broken into cross-entropy, weight norm, and prox reg
%        components (ceCost, wCost, pCost)

%% default values
po = false;
if exist('pred_only','var')
  po = pred_only;
end;

%% reshape into network
stack = params2stack(theta, ei);
numHidden = numel(ei.layer_sizes) - 1;
hAct = cell(numHidden+2, 1);
gradStack = cell(numHidden+1, 1);

%% forward prop

%%% YOUR CODE HERE %%%
cost = 0;
m = size(data,2); % Number of samples
K = size(stack{numHidden+1}.W, 1); % Number of classes
D = size(data, 1); % Number of dimensions (length of input vector)

labels_nice = zeros(m, K);
labels_ind = sub2ind(size(labels_nice), 1:m, labels');
labels_nice(labels_ind) = 1;

nl = numHidden+2;
pred_prob = zeros(m, K);
Delta_W = cell(nl, 1);
Delta_b = cell(nl, 1);

hAct{1} = zeros(m, D);
for l=1:numHidden+1
    Delta_W{l} = zeros(size(stack{l}.W));
    Delta_b{l} = zeros(size(stack{l}.b));
    hAct{l+1} = zeros(m, size(stack{l}.W, 1)); % Space for the thing
end

for i = 1:m % Loop over all data items
    x = data(:,i);
    hAct{1}(i,:) = x;
    % Forward pass
    for l = 1:numHidden
        % Compute the activations based on inputs
        % z^(l+1) = W^l * a^l + b^l [ Sum of weighted inputs ]
        % a^(l+1) = f(z^(l+1)) [ activation function (in this case,
        % logistic) ]

        W_l = stack{l}.W;
        b_l = stack{l}.b;
        a_l = hAct{l}(i,:)';

        z = W_l*a_l + b_l;
        hAct{l+1}(i,:) = sigmoid(z);
    end

    % Compute the output layer
    W = stack{nl-1}.W;
    b = stack{nl-1}.b;

    numerator = exp(W*hAct{nl-1}(i,:)'+b);
    denominator = sum(numerator);
    probs = numerator./denominator;
    pred_prob(i,:) = probs;
    hAct{nl}(i,:) = probs;

end %% End loop over m data points
%% return here if only predictions desired.
if po
  cost = -1; ceCost = -1; wCost = -1; numCorrect = -1;
  grad = [];
  return;
end;

%% compute cost
%%% YOUR CODE HERE %%%
cost = 0;
for i=1:m
    cost = cost - log(pred_prob(i, labels(i)));
end
% cost = cost - log(pred_prob(i, labels(i)));


%% compute gradients using backpropagation
%%% YOUR CODE HERE %%%

% At the end of forwardprop, hAct{numHidden+2} = the final
% activations (non-hidden).
% Backprop
% δ(nl)=−∑i=1m[(1{y(i)=k}−P(y(i)=k|x(i);θ))]
% δ(l)=((W(l))Tδ(l+1))∙f′(z(l))
% Compute the desired partial derivatives:
% ∇W(l)J(W,b;x,y) = δ(l+1)(a(l))T
% ∇b(l)J(W,b;x,y) = δ(l+1)
delta_l = cell(nl,1);
% From http://ufldl.stanford.edu/tutorial/supervised/ExerciseSupervisedNeuralNetwork/
delta_l{nl} = -(labels_nice - pred_prob);

for l = nl-1:-1:1
    W_l = stack{l}.W;
    % b_l = stack{l}.b;
    % delta_l{l} = W_l'*(delta_l{l+1}).*(hAct{l+1}.*(1-hAct{l+1}));
    for i = 1:m
        a_l = hAct{l}(i, :)';
        % This line had issues
        delta_l{l}(i,:) = (W_l'*(delta_l{l+1}(i,:)')).*a_l.*(1-a_l);

        % Compute gradients and stuff
        del = delta_l{l+1}(i,:)'*a_l';
        Delta_W{l} = Delta_W{l} + del;
        Delta_b{l} = Delta_b{l} + delta_l{l+1}(i,:)';
    end
end

% for l = nl-1:-1:1
%     Delta_W{l} = delta_l{l+1}*hAct{l}';
%     Delta_b{l} = delta_l{l+1};
% end
% Use Delta_W and Delta_b to compute gradients
for l = 1:numHidden+1
    gradStack{l}.W = Delta_W{l};
    gradStack{l}.b = Delta_b{l};
end

%% compute weight penalty cost and gradient for non-bias terms
%%% YOUR CODE HERE %%%

%% reshape gradients into vector
[grad] = stack2params(gradStack);
end
