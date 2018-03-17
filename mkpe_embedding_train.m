function state = mkpe_embedding_train(K_c, K_x, K_z, parameters)
    rand('state', parameters.seed); %#ok<RAND>
    randn('state', parameters.seed); %#ok<RAND>

    N_x = size(K_x, 1);
    N_z = size(K_z, 1);
    R = parameters.R;
    sigma_e = parameters.sigma_e;
    epsilon = parameters.epsilon;
    lambda_c = parameters.lambda_c;
    lambda_x = parameters.lambda_x;
    lambda_z = parameters.lambda_z;
    iteration = parameters.iteration;
    learn_sigma_e = parameters.learn_sigma_e;

    indices_c = find(~isnan(K_c));
    count_c = length(indices_c);
    indices_x = find(~isnan(K_x));
    count_x = length(indices_x);
    indices_z = find(~isnan(K_z));
    count_z = length(indices_z);

    gamma_x = 1;
    gamma_z = 1;
    gamma_eta = 1;

    E_x = project_to_stiefel_manifold(randn(N_x, R));
    E_z = project_to_stiefel_manifold(randn(N_z, R));
    DE_c = pdist2(E_x, E_z).^2;
    KE_c = exp(-DE_c / sigma_e^2);
    DE_x = pdist2(E_x, E_x).^2;
    KE_x = exp(-DE_x / sigma_e^2);
    DE_z = pdist2(E_z, E_z).^2;
    KE_z = exp(-DE_z / sigma_e^2);
    objective_c = sum(sum((KE_c(indices_c) - K_c(indices_c)).^2)) / count_c;
    objective_x = sum(sum((KE_x(indices_x) - K_x(indices_x)).^2)) / count_x;
    objective_z = sum(sum((KE_z(indices_z) - K_z(indices_z)).^2)) / count_z;
    objective = lambda_c * objective_c + lambda_x * objective_x + lambda_z * objective_z;

    iter = 0;
    fprintf(1, 'iteration=%3d objective=%.6f\n', iter, objective);
    while 1
        iter = iter + 1;
        return_x = 0;
        return_z = 0;
        return_eta = 0;

        E_x_gradient = zeros(N_x, R);
        for i = 1:N_x
            j = ~isnan(K_c(i, :));
            E_x_gradient(i, :) = E_x_gradient(i, :) - 4 * lambda_c * sum(bsxfun(@times, KE_c(i, j)' .* (KE_c(i, j) - K_c(i, j))', bsxfun(@minus, E_x(i, :), E_z(j, :))), 1) / sigma_e.^2 / count_c;
        end
        if lambda_x ~= 0
            for i = 1:N_x
                j = ~isnan(K_x(i, :));
                E_x_gradient(i, :) = E_x_gradient(i, :) - 8 * lambda_x * sum(bsxfun(@times, KE_x(i, j)' .* (KE_x(i, j) - K_x(i, j))', bsxfun(@minus, E_x(i, :), E_x(j, :))), 1) / sigma_e.^2 / count_x;
            end
        end
        E_x_gradient = (E_x * E_x_gradient' * E_x - E_x_gradient);
        E_x_gradient_norm = sum(diag(E_x_gradient' * (eye(N_x, N_x) - 0.5 * (E_x * E_x')) * E_x_gradient));
        if sqrt(E_x_gradient_norm) < epsilon
            return_x = 1;
            objective_c = [objective_c, objective_c(end)]; %#ok<AGROW>
            objective_x = [objective_x, objective_x(end)]; %#ok<AGROW>
            objective_z = [objective_z, objective_z(end)]; %#ok<AGROW>
            objective = [objective, objective(end)]; %#ok<AGROW>
        else
            while 1
                E_x_new = E_x + 2 * gamma_x * E_x_gradient;
                E_x_new = project_to_stiefel_manifold(E_x_new);
                DE_c = pdist2(E_x_new, E_z).^2;
                KE_c = exp(-DE_c / sigma_e^2);
                DE_x = pdist2(E_x_new, E_x_new).^2;
                KE_x = exp(-DE_x / sigma_e^2);
                objective_new = lambda_c * sum(sum((KE_c(indices_c) - K_c(indices_c)).^2)) / count_c + lambda_x * sum(sum((KE_x(indices_x) - K_x(indices_x)).^2)) / count_x + lambda_z * sum(sum((KE_z(indices_z) - K_z(indices_z)).^2)) / count_z;
                if objective(end) - objective_new >= gamma_x * E_x_gradient_norm
                    gamma_x = 2 * gamma_x;
                else
                    break;
                end
            end

            while 1
                E_x_new = E_x + gamma_x * E_x_gradient;
                E_x_new = project_to_stiefel_manifold(E_x_new);
                DE_c = pdist2(E_x_new, E_z).^2;
                KE_c = exp(-DE_c / sigma_e^2);
                DE_x = pdist2(E_x_new, E_x_new).^2;
                KE_x = exp(-DE_x / sigma_e^2);
                objective_new = lambda_c * sum(sum((KE_c(indices_c) - K_c(indices_c)).^2)) / count_c + lambda_x * sum(sum((KE_x(indices_x) - K_x(indices_x)).^2)) / count_x + lambda_z * sum(sum((KE_z(indices_z) - K_z(indices_z)).^2)) / count_z;
                if objective(end) - objective_new < 0.5 * gamma_x * E_x_gradient_norm
                    gamma_x = 0.5 * gamma_x;
                else
                    break;
                end
            end

            E_x = E_x + gamma_x * E_x_gradient;
            E_x = project_to_stiefel_manifold(E_x);
            DE_c = pdist2(E_x, E_z).^2;
            KE_c = exp(-DE_c / sigma_e^2);
            DE_x = pdist2(E_x, E_x).^2;
            KE_x = exp(-DE_x / sigma_e^2);
            objective_c_last = sum(sum((KE_c(indices_c) - K_c(indices_c)).^2)) / count_c;
            objective_x_last = sum(sum((KE_x(indices_x) - K_x(indices_x)).^2)) / count_x;
            objective_z_last = sum(sum((KE_z(indices_z) - K_z(indices_z)).^2)) / count_z;
            objective_last = lambda_c * objective_c_last + lambda_x * objective_x_last + lambda_z * objective_z_last;
            objective_c = [objective_c, objective_c_last]; %#ok<AGROW>
            objective_x = [objective_x, objective_x_last]; %#ok<AGROW>
            objective_z = [objective_z, objective_z_last]; %#ok<AGROW>
            objective = [objective, objective_last]; %#ok<AGROW>
            
            fprintf(1, 'iteration=%3d objective=%.6f norm_x=%.6f gamma_x=%.6f\n', iter, objective_last, sqrt(E_x_gradient_norm), gamma_x);
        end
        
        E_z_gradient = zeros(N_z, R);
        for j = 1:N_z
            i = ~isnan(K_c(:, j));
            E_z_gradient(j, :) = E_z_gradient(j, :) - 4 * lambda_c * sum(bsxfun(@times, KE_c(i, j) .* (KE_c(i, j) - K_c(i, j)), bsxfun(@minus, E_z(j, :), E_x(i, :))), 1) / sigma_e.^2 / count_c;
        end
        if lambda_z ~= 0
            for i = 1:N_z
                j = ~isnan(K_z(i, :));
                E_z_gradient(i, :) = E_z_gradient(i, :) - 8 * lambda_z * sum(bsxfun(@times, KE_z(i, j)' .* (KE_z(i, j) - K_z(i, j))', bsxfun(@minus, E_z(i, :), E_z(j, :))), 1) / sigma_e.^2 / count_z;
            end
        end
        E_z_gradient = (E_z * E_z_gradient' * E_z - E_z_gradient);
        E_z_gradient_norm = sum(diag(E_z_gradient' * (eye(N_z, N_z) - 0.5 * (E_z * E_z')) * E_z_gradient));
        if sqrt(E_z_gradient_norm) < epsilon
            return_z = 1;
            objective_c = [objective_c, objective_c(end)]; %#ok<AGROW>
            objective_x = [objective_x, objective_x(end)]; %#ok<AGROW>
            objective_z = [objective_z, objective_z(end)]; %#ok<AGROW>
            objective = [objective, objective(end)]; %#ok<AGROW>
        else
            while 1
                E_z_new = E_z + 2 * gamma_z * E_z_gradient;
                E_z_new = project_to_stiefel_manifold(E_z_new);
                DE_c = pdist2(E_x, E_z_new).^2;
                KE_c = exp(-DE_c / sigma_e^2);
                DE_z = pdist2(E_z_new, E_z_new).^2;
                KE_z = exp(-DE_z / sigma_e^2);
                objective_new = lambda_c * sum(sum((KE_c(indices_c) - K_c(indices_c)).^2)) / count_c + lambda_x * sum(sum((KE_x(indices_x) - K_x(indices_x)).^2)) / count_x + lambda_z * sum(sum((KE_z(indices_z) - K_z(indices_z)).^2)) / count_z;
                if objective(end) - objective_new >= gamma_z * E_z_gradient_norm
                    gamma_z = 2 * gamma_z;
                else
                    break;
                end
            end

            while 1
                E_z_new = E_z + gamma_z * E_z_gradient;
                E_z_new = project_to_stiefel_manifold(E_z_new);
                DE_c = pdist2(E_x, E_z_new).^2;
                KE_c = exp(-DE_c / sigma_e^2);
                DE_z = pdist2(E_z_new, E_z_new).^2;
                KE_z = exp(-DE_z / sigma_e^2);
                objective_new = lambda_c * sum(sum((KE_c(indices_c) - K_c(indices_c)).^2)) / count_c + lambda_x * sum(sum((KE_x(indices_x) - K_x(indices_x)).^2)) / count_x + lambda_z * sum(sum((KE_z(indices_z) - K_z(indices_z)).^2)) / count_z;
                if objective(end) - objective_new < 0.5 * gamma_z * E_z_gradient_norm
                    gamma_z = 0.5 * gamma_z;
                else
                    break;
                end
            end

            E_z = E_z + gamma_z * E_z_gradient;
            E_z = project_to_stiefel_manifold(E_z);
            DE_c = pdist2(E_x, E_z).^2;
            KE_c = exp(-DE_c / sigma_e^2);
            DE_z = pdist2(E_z, E_z).^2;
            KE_z = exp(-DE_z / sigma_e^2);
            objective_c_last = sum(sum((KE_c(indices_c) - K_c(indices_c)).^2)) / count_c;
            objective_x_last = sum(sum((KE_x(indices_x) - K_x(indices_x)).^2)) / count_x;
            objective_z_last = sum(sum((KE_z(indices_z) - K_z(indices_z)).^2)) / count_z;
            objective_last = lambda_c * objective_c_last + lambda_x * objective_x_last + lambda_z * objective_z_last;
            objective_c = [objective_c, objective_c_last]; %#ok<AGROW>
            objective_x = [objective_x, objective_x_last]; %#ok<AGROW>
            objective_z = [objective_z, objective_z_last]; %#ok<AGROW>
            objective = [objective, objective_last]; %#ok<AGROW>

            fprintf(1, 'iteration=%3d objective=%.6f norm_z=%.6f gamma_z=%.6f\n', iter, objective_last, sqrt(E_z_gradient_norm), gamma_z);
        end

        if learn_sigma_e == 1
            eta_gradient = 0;
            eta_gradient = eta_gradient - 4 * lambda_c * sum(KE_c(indices_c) .* (KE_c(indices_c) - K_c(indices_c)) .* DE_c(indices_c)) / sigma_e.^2 / count_c;
            eta_gradient = eta_gradient - 4 * lambda_x * sum(KE_x(indices_x) .* (KE_x(indices_x) - K_x(indices_x)) .* DE_x(indices_x)) / sigma_e.^2 / count_x;
            eta_gradient = eta_gradient - 4 * lambda_z * sum(KE_z(indices_z) .* (KE_z(indices_z) - K_z(indices_z)) .* DE_z(indices_z)) / sigma_e.^2 / count_z;
            eta_gradient_norm = eta_gradient^2;
            if sqrt(eta_gradient_norm) < epsilon
                return_eta = 1;
                objective_c = [objective_c, objective_c(end)]; %#ok<AGROW>
                objective_x = [objective_x, objective_x(end)]; %#ok<AGROW>
                objective_z = [objective_z, objective_z(end)]; %#ok<AGROW>
                objective = [objective, objective(end)]; %#ok<AGROW>
            else
                while 1
                    sigma_e_new = exp(log(sigma_e) + 2 * gamma_eta * eta_gradient);
                    KE_c = exp(-DE_c / sigma_e_new^2);
                    KE_x = exp(-DE_x / sigma_e_new^2);
                    KE_z = exp(-DE_z / sigma_e_new^2);
                    objective_new = lambda_c * sum(sum((KE_c(indices_c) - K_c(indices_c)).^2)) / count_c + lambda_x * sum(sum((KE_x(indices_x) - K_x(indices_x)).^2)) / count_x + lambda_z * sum(sum((KE_z(indices_z) - K_z(indices_z)).^2)) / count_z;
                    if objective(end) - objective_new >= gamma_eta * eta_gradient_norm
                        gamma_eta = 2 * gamma_eta;
                    else
                        break;
                    end
                end

                while 1
                    sigma_e_new = exp(log(sigma_e) + gamma_eta * eta_gradient);
                    KE_c = exp(-DE_c / sigma_e_new^2);
                    KE_x = exp(-DE_x / sigma_e_new^2);
                    KE_z = exp(-DE_z / sigma_e_new^2);
                    objective_new = lambda_c * sum(sum((KE_c(indices_c) - K_c(indices_c)).^2)) / count_c + lambda_x * sum(sum((KE_x(indices_x) - K_x(indices_x)).^2)) / count_x + lambda_z * sum(sum((KE_z(indices_z) - K_z(indices_z)).^2)) / count_z;
                    if objective(end) - objective_new < 0.5 * gamma_eta * eta_gradient_norm
                        gamma_eta = 0.5 * gamma_eta;
                    else
                        break;
                    end
                end

                sigma_e = exp(log(sigma_e) + gamma_eta * eta_gradient);
                KE_c = exp(-DE_c / sigma_e_new^2);
                KE_x = exp(-DE_x / sigma_e_new^2);
                KE_z = exp(-DE_z / sigma_e_new^2);
                objective_c_last = sum(sum((KE_c(indices_c) - K_c(indices_c)).^2)) / count_c;
                objective_x_last = sum(sum((KE_x(indices_x) - K_x(indices_x)).^2)) / count_x;
                objective_z_last = sum(sum((KE_z(indices_z) - K_z(indices_z)).^2)) / count_z;
                objective_last = lambda_c * objective_c_last + lambda_x * objective_x_last + lambda_z * objective_z_last;
                objective_c = [objective_c, objective_c_last]; %#ok<AGROW>
                objective_x = [objective_x, objective_x_last]; %#ok<AGROW>
                objective_z = [objective_z, objective_z_last]; %#ok<AGROW>
                objective = [objective, objective_last]; %#ok<AGROW>

                fprintf(1, 'iteration=%3d objective=%.6f norm_e=%.6f gamma_e=%.6f\n', iter, objective_last, sqrt(eta_gradient_norm), gamma_eta);
            end
        else
            return_eta = 1;
            objective_c = [objective_c, objective_c(end)]; %#ok<AGROW>
            objective_x = [objective_x, objective_x(end)]; %#ok<AGROW>
            objective_z = [objective_z, objective_z(end)]; %#ok<AGROW>
            objective = [objective, objective(end)]; %#ok<AGROW>
        end

        if return_x == 1 && return_z == 1 && return_eta == 1
            break;
        end
        if iter == iteration
           break;
        end
    end

    state.E_x = E_x;
    state.E_z = E_z;
    state.sigma_e = sigma_e;
    state.objective_c = objective_c;
    state.objective_x = objective_x;
    state.objective_z = objective_z;
    state.objective = objective;
end

function Q = project_to_stiefel_manifold(Q)
    [U, ~, V] = svd(Q);
    Q = U * eye(size(Q)) * V';
end
