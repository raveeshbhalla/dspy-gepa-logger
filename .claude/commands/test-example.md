When we make significant changes to the codebase, we want to make sure everything is running as intended. To do this we follow the following process:

1. Check if we have .env.local file set up properly in the examples folder. If not, warn the user.
2. Run the examples/eg_v2_simple.py with the server attached
3. We validate if the optimization is running fine, and fix any errors in the example itself
4. We validate that the server is receiving all the logs correctly. This includes:
   - Is it getting raw logs itself?
   - Are the logs sufficiently detailed? They should be near identical to the terminal logs
   - Are the iterations being updated correctly and can be seen in the iterations tab? To do that, we validate that the APIs that power the front-end are returning information correctly
    - Is the information in each iteration complete and accurate? Are we seeing all the prompt changes (parent and child prompts), are we seeing the reflection data, are we seeing all the performance comparison?
    - Are the performance comparisons complete? Do they have the input information, the scores for both the parent and child prompts, the feedback for the parent and child prompt, structured responses for both the parent and child prompts?
5. Is the lineage being updated correctly? Are we able to compare two prompts in the lineage and see their performance comparison (as detailed as in the iterations tab)?
6. Is the overview being updated correctly? Are we able to see all the data, the prompt comparisons and performance comparisons as detailed as the iterations tab?

If we see any of these issues, we should inform the user, start identifying fixes, and start working on them.
