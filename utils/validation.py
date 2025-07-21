def validate_input(input_data):
    required_fields = [
        'Time_spent_Alone',
        'Social_event_attendance',
        'Going_outside',
        'Friends_circle_size',
        'Post_frequency',
        'Stage_fear',
        'Drained_after_socializing'
    ]
    
    # Check all required fields are present
    if not all(field in input_data for field in required_fields):
        return False
    
    # Validate numerical fields
    numerical_fields = [
        'Time_spent_Alone',
        'Social_event_attendance',
        'Going_outside',
        'Friends_circle_size',
        'Post_frequency'
    ]
    
    for field in numerical_fields:
        try:
            float(input_data[field])
        except (TypeError, ValueError):
            return False
    
    # Validate categorical fields
    categorical_fields = {
        'Stage_fear': ['Yes', 'No'],
        'Drained_after_socializing': ['Yes', 'No']
    }
    
    for field, allowed_values in categorical_fields.items():
        if input_data[field] not in allowed_values:
            return False
    
    return True