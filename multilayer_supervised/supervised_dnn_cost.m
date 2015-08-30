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

m = size(data,2);
labels_nice = zeros(length(labels), max(labels));
labels_nice(:, labels) = 1;
nl = numHidden+2;
Delta_W = cell(nl, 1);
Delta_b = cell(nl, 1);

for i = 1:m % Loop over all data items
    x = data(:,i);
    hAct{1} = x;
    % Forward pass
    for l = 1:numHidden
        % Compute the activations based on inputs
        % z^(l+1) = W^l * a^l + b^l [ Sum of weighted inputs ]
        % a^(l+1) = f(z^(l+1)) [ activation function (in this case,
        % logistic) ]

        W_l = stack{l}.W;
        b_l = stack{l}.b;
        a_l = hAct{l};

        z = W_l*a_l + b_l;
        hAct{l+1} = sigmoid(z);
    end
    
    % Compute the output layer
    W = stack{nl-1}.W;
    b = stack{nl-1}.b;

    numerator = exp(W*hAct{nl-1}+b);
    denominator = sum(numerator);
    probs = numerator./denominator;
    hAct{nl} = probs;
    % At the end of forwardprop, hAct{numHidden+2} = the final
    % activations (non-hidden).
    % Backprop
    % δ(nl)=−∑i=1m[(1{y(i)=k}−P(y(i)=k|x(i);θ))]
    % δ(l)=((W(l))Tδ(l+1))∙f′(z(l))
    % Compute the desired partial derivatives:
    % ∇W(l)J(W,b;x,y) = δ(l+1)(a(l))T
    % ∇b(l)J(W,b;x,y) = δ(l+1)

    delta_l = cell(nl,1);
    
    delta_l{nl} = -(labels_nice(i,:)' - probs); %% From http://ufldl.stanford.edu/tutorial/supervised/ExerciseSupervisedNeuralNetwork/
    % Delta_W{nl} = Delta_W{nl} + delta_l{nl}*hAct{nl}';
    % Delta_b{nl} = Delta_b{nl} + delta_l{nl};
    for l = nl-1:-1:1
        W_l = stack{l}.W;
        b_l = stack{l}.b;
        a_l = hAct{l+1};

        % This line has issues
        % (256x784)' x (10x1) x (784x1) .x (784x1)
        delta_l{l} = W_l'*(delta_l{l+1}.*a_l.*(1-a_l));

        % TODO: Compute gradients and stuff
        Delta_W{l} = Delta_W{l} + delta_l{l+1}*hAct{l}';
        Delta_b{l} = Delta_b{l} + delta_l{l+1};
    end

end %% End loop over m data points
%% return here if only predictions desired.
if po
  cost = -1; ceCost = -1; wCost = -1; numCorrect = -1;
  grad = [];
  return;
end;

%% compute cost
%%% YOUR CODE HERE %%%

%% compute gradients using backpropagation
%%% YOUR CODE HERE %%%

% Use Delta_W and Delta_b to compute gradients
for l = 1:numHidden
    grad{l}.W = Delta_W{l};
    grad{l}.b = Delta_b{l};
end

%% compute weight penalty cost and gradient for non-bias terms
%%% YOUR CODE HERE %%%

%% reshape gradients into vector
[grad] = stack2params(gradStack);
end



