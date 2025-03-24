SELECT SNOWFLAKE.CORTEX.COMPLETE(
    'llama2-70b-chat',
    [
        {
            'role': 'user',
            'content': 'how does a snowflake get its unique pattern?'
        }
    ],
    {
        'temperature': 0.7,
        'max_tokens': 10
    }
);
