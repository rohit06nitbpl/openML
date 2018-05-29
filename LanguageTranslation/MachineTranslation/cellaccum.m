function [x y b] = cellaccum(hin,din,hout,delta)
  g = hout.*(1-hout).*delta;
  x = g*hin';
  y = g*din';
  b = g;
end