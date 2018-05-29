function [y b] = softmaxaccum(del,hin)
  y = del*hin';
  b = del;
end
