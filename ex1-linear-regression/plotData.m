function plotData(x, y)

  figure; % open a new figure window

  plot(x, y, 'r*', 'MarkerSize', 5);
  ylabel('Profit in $10,000s');
  xlabel('Population of city in 10,000s');

end
