% function y(x)
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

% for different number of neurons
for num_neurons = neuron_list
    net = fitnet(num_neurons, 'traingdx');  
    net.divideFcn = 'dividetrain';  % Use all data for training
    net.trainParam.epochs = 1;  % Only one epoch at a time in sequential mode
    
    % train
    net = configure(net, x_train, y_train);  % Configure the network based on training data
    for epoch = 1:100
        disp(epoch)
        for i = 1:numel(x_train)
            xi = x_train(i);
            yi = y_train(i);
            
            net = adapt(net, xi, yi);  % Sequential update using adapt
        end
    end
    
    % Test on the entire test set
    y_pred = net(x_test);

    % Calculate MSE for both training and testing data
    mse_train = perform(net, y_train, net(x_train));
    mse_test = perform(net, y_test, y_pred);
    
    % Save the results
    mse_train_results = [mse_train_results; mse_train];
    mse_test_results = [mse_test_results; mse_test];
    
    % Display results
    fprintf('Number of Neurons: %d\n', num_neurons);
    fprintf('Training MSE: %.4f\n', mse_train);
    fprintf('Test MSE: %.4f\n', mse_test);
    
    % Plot the predictions vs true function
    figure;
    plot(x_test, y_test, 'b', 'LineWidth', 1.5);
    hold on;
    plot(x_test, y_pred, 'r--', 'LineWidth', 1.5); 
    legend('True Function', 'MLP Prediction');
    title(sprintf('MLP with %d Neurons', num_neurons));
    xlabel('x');
    ylabel('y');
    saveas(gcf, sprintf('./Q2a/MLP_with_%d_Neurons.png', num_neurons));
    hold off;
    
    % Extrapolation on x = -3 and x = 3
    y_pred_extrap = net([-3, 3]);
    fprintf('MLP prediction for x = -3: %.4f\n', y_pred_extrap(1));
    fprintf('MLP prediction for x = 3: %.4f\n', y_pred_extrap(2));
    
    % Check if the network is overfitting or underfitting
    if mse_test > mse_train * 2  
        fprintf('The network may be overfitting with %d neurons.\n', num_neurons);
    elseif mse_train > 0.1 && mse_test > 0.1 
        fprintf('The network may be underfitting with %d neurons.\n', num_neurons);
    else
        fprintf('The network is fitting well with %d neurons.\n', num_neurons);
    end
end

% Plot MSE vs Number of Neurons
figure;
plot(neuron_list, mse_train_results, '-o', 'LineWidth', 1.5);
hold on;
plot(neuron_list, mse_test_results, '-x', 'LineWidth', 1.5);
legend('Training MSE', 'Test MSE');
title('MSE vs Number of Neurons');
xlabel('Number of Neurons');
ylabel('MSE');
saveas(gcf, 'MSE_vs_Number_of_Neurons.png');
hold off;
