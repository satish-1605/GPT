
def count_parameters(model)-> int:
    """
    count the total number of trainable parameter
    """
    return sum(parameter.numel() 
               for parameter in model.parameters() 
               if parameter.requires_grad)


def count_parameters_millions(model) -> float:
    """
    Return trainable parameters in millions.
    """
    return count_parameters(model) / 1e6


    
