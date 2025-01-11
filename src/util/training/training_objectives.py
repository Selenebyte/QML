import torch


def target_function(x, coeffs, device, reverse_conj_coef=False):
    """Compute the target function for input and coefficients

    Computes the target fourier series of the input and coefficients.

    Args:
                    x: Input data.
        coeffs: Tensor of fourier coefficients in the form {zeroth coefficient, non-conjugates, conjugates}.
        device: Device for the torch tensors.

    returns:
          Tensor of the real part of the output values.

    Raises:
                    N/A
    """
    n = torch.as_tensor(coeffs.shape[0]).to(device)
    res = torch.clone(coeffs[0]) * torch.ones(x.shape).to(device)
    for idx in range(0, int(torch.floor(n / 2))):
        exponent = (idx + 1) * x * 1j
        idx += 1
        if reverse_conj_coef:
            res += coeffs[idx] * torch.exp(exponent) + coeffs[-idx] * torch.exp(
                -exponent
            )
        else:
            res += coeffs[idx] * torch.exp(exponent) + coeffs[
                int(torch.floor(n / 2)) + idx
            ] * torch.exp(-exponent)

    return torch.real(res)


def predictions_from_target_function(
    x, fourier_coef_list, device, reverse_conj_coef=False
):
    """Computes the predictions of the input and a list of fourier coefficients.

    Computes the predictions of the input and a list of fourier coefficients from the target function.

    Args:
                    x: Input data.
        fourier_coef_list: Tensor of the target functions fourier coefficients.
        device: Device for the torch tensors.

    returns:
          Tensor of the predictions of target functions.

    Raises:
                    N/A
    """
    y_predictions_for_coefficients = []

    for fourier_coef in fourier_coef_list:
        fourier_coef = torch.tensor(fourier_coef).to(device)
        y_predictions_for_coefficients.append(
            target_function(x, fourier_coef, device, reverse_conj_coef)
        )

    return y_predictions_for_coefficients


def loss_of_fourier_coefficients(fourier_coef, inputs, target_y, loss_fn, device):
    """Computes the loss of the input data.

    Computes the loss of the input data using the loss function with the target y values.

    Args:
                    fourier_coef: Fourier coefficients for target function.
        inputs: Input data.
        target_y: Target y values for loss.
        loss_fn: Loss function used.
        device: Device for the torch tensors.

    returns:
          Tensor of the predictions of target functions.

    Raises:
                    N/A
    """
    predictions = target_function(inputs, fourier_coef, device)
    return loss_fn(predictions, target_y)
