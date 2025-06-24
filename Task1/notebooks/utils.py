def calculate_feature_selection_score(selected_features, all_relevant_features, all_irrelevant_features):
    """
    Calculate the feature selection score as described by Bolón-Canedo et al. (2012).
    
    The score evaluates the quality of feature selection by considering both:
    - The inclusion of relevant features
    - The exclusion of irrelevant features
    
    Parameters:
    -----------
    selected_features : list or set
        The features selected by the feature selection method.
    all_relevant_features : list or set
        The complete set of relevant features (ground truth).
    all_irrelevant_features : list or set
        The complete set of irrelevant features (ground truth).
    alpha : float, default=0.5
        The weight parameter that balances the importance between including
        relevant features and excluding irrelevant ones. Default is 0.5,
        giving equal weight to both aspects.
    
    Returns:
    --------
    float
        The success score (0-100). Higher values indicate better feature selection.
        A score of 100 means perfect selection of all relevant features and
        exclusion of all irrelevant features.
    """
    # Convert inputs to sets for easier operations
    selected_features = set(selected_features)
    all_relevant_features = set(all_relevant_features)
    all_irrelevant_features = set(all_irrelevant_features)
    
    # Calculate the number of relevant features selected (Rs)
    Rs = len(selected_features.intersection(all_relevant_features))
    
    # Calculate the total number of relevant features (Rt)
    Rt = len(all_relevant_features)
    
    # Calculate the number of irrelevant features selected (Is)
    Is = len(selected_features.intersection(all_irrelevant_features))
    
    # Calculate the total number of irrelevant features (It)
    It = len(all_irrelevant_features)
    
    # Calculate the success score using the formula from Bolón-Canedo et al. (2012)
    # Suc. = (Rs/Rt - α*Is/It) × 100
    # Handle edge cases to avoid division by zero
    if Rt == 0:
        relevant_ratio = 0
    else:
        relevant_ratio = Rs / Rt
        
    if It == 0:
        irrelevant_ratio = 0
    else:
        irrelevant_ratio = Is / It
    alpha = min(0.5, Rt/It)
    success_score = (relevant_ratio - alpha * irrelevant_ratio) * 100
    
    return max(0, success_score)  # Ensure the score is not negative