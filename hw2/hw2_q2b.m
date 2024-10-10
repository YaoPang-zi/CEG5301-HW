% functioin y(x)
y = @(x) 1.2 * sin(pi * x) - cos(2.4 * pi * x);

% train/test data
x_train = -2:0.05:2;
y_train = y(x_train);

x_test = -2:0.01:2;
y_test = y(x_test);

% history of train and test results
mse_train_results = [];
mse_test_results = [];
neuron_list = [1:10, 20, 50, 100];

% for different number of neuron
for num_neurons = neuron_list
    net = fitnet(num_neurons, 'trainlm');
    net.divideFcn = 'dividetrain';
    net.trainParam.epochs = 100;
    net.divideParam.trainRatio = 100/100;
    
    % train
    [net, tr] = train(net, x_train, y_train);
    
    % test
    y_pred = net(x_test);

    % MSE
    mse_train = perform(net, y_train, net(x_train));
    mse_test = perform(net, y_test, y_pred);
    
    % results
    mse_train_results = [mse_train_results; mse_train];
    mse_test_results = [mse_test_results; mse_test];
    
    fprintf('Number of Neurons: %d\n', num_neurons);
    fprintf('Training MSE: %.4f\n', mse_train);
    fprintf('Test MSE: %.4f\n', mse_test);
    
    figure;
    plot(x_test, y_test, 'b', 'LineWidth', 1.5);
    hold on;
    plot(x_test, y_pred, 'r--', 'LineWidth', 1.5); 
    legend('True Function', 'MLP Prediction');
    title(sprintf('MLP with %d Neurons', num_neurons));
    xlabel('x');
    ylabel('y');
    saveas(gcf, sprintf('./Q2b/MLP_with_%d_Neurons.png', num_neurons));
    hold off;
    
    % predict on -3 and 3
    y_pred_extrap = net([-3, 3]);
    fprintf('MLP prediction for x = -3: %.4f\n', y_pred_extrap(1));
    fprintf('MLP prediction for x = 3: %.4f\n', y_pred_extrap(2));
    
    % check if overfitting or underfitting
    if mse_test > mse_train * 2  
        fprintf('The network may be overfitting with %d neurons.\n', num_neurons);
    elseif mse_train > 0.01 && mse_test > 0.01 
        fprintf('The network may be underfitting with %d neurons.\n', num_neurons);
    else
        fprintf('The network is fitting well with %d neurons.\n', num_neurons);
    end
end

% mse for different number of neuron
figure;
plot(neuron_list, mse_train_results, '-o', 'LineWidth', 1.5);
hold on;
plot(neuron_list, mse_test_results, '-x', 'LineWidth', 1.5);
legend('Training MSE', 'Test MSE');
title('MSE vs Number of Neurons');
xlabel('Number of Neurons');
ylabel('MSE');
saveas(gcf, './Q2b/MSE_vs_Number_of_Neurons.png');
hold off;
