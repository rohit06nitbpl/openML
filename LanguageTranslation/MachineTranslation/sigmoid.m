function g = sigmoid(z)
  %SIGMOID Compute sigmoid functoon
  %   J = SIGMOID(z) computes the sigmoid of z.
  % z can be a scalar, a vector, or a matrix.
  g = zeros(size(z));
  g = 1 ./ (1 + exp(-z));
end