function y = softmaxactivate(wvh,h,b)
  g = wvh*h + b;
  g = exp(g);
  s = sum(g);
  y = g ./ s;
end