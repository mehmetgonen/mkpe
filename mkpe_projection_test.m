function prediction = mkpe_projection_test(X, Z, state)
    prediction.E_x = X * state.Q_x;
    prediction.E_z = Z * state.Q_z;
end
