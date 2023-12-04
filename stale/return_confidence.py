import requests

def get_chatgpt_answer(question, context=None, max_length=100, temperature=0.7, top_p=0.9, confidence_threshold=0.8):
    """
    Calls the ChatGPT API to retrieve a predicted answer for a given question.
    Returns the predicted answer and its confidence score, if it meets the specified threshold.
    """
    # Set up the API endpoint and query parameters
    url = 'https://api.openai.com/v1/chatgpt/answer'
    headers = {'Authorization': 'Bearer YOUR_API_KEY'}
    params = {
        'question': question,
        'max_length': max_length,
        'temperature': temperature,
        'top_p': top_p
    }
    if context:
        params['context'] = context
    
    # Call the API and retrieve the response
    response = requests.get(url, headers=headers, params=params)
    response.raise_for_status()
    data = response.json()
    
    # Check if the predicted answer meets the confidence threshold
    if data['confidence'] >= confidence_threshold:
        return data['answer'], data['confidence']
    else:
        return None, 0.0
