# instruction for training data
# from self-rag's repo
TASK_INST_TRAIN =   {
                    "wow": "Given a chat history separated by new lines, generates an informative, knowledgeable and engaging response. ",
                    "fever": "Is the following statement correct or not? Say true if it's correct; otherwise say false.",
                    "eli5": "Provide a paragraph-length response using simple words to answer the following question.",
                    "obqa": "Given four answer candidates, A, B, C and D, choose the best answer choice.",
                    "arc_easy": "Given four answer candidates, A, B, C and D, choose the best answer choice.",
                    "arc_c": "Given four answer candidates, A, B, C and D, choose the best answer choice.",
                    "trex": "Given the input format 'Subject Entity [SEP] Relationship Type,' predict the target entity.",
                    "asqa": "Answer the following question. The question may be ambiguous and have multiple correct answers, and in that case, you have to provide a long-form answer including all correct answers."
                    }

# instruction for eval data
# these instructions are use for close-domain, other datasets like popqa will use the original questions as prompt. 
# from self-rag's paper
TASK_INST_EVAL =    {
                    "arc": "Given four answer candidates, A, B, C and D, choose the best answer choice. Please answer with the capitalized alphabet only, without adding any extra phrase or period.",
                    "pubhealth": "Is the following statement correct or not? Say true if it’s correct; otherwise, say false. Don’t capitalize or add periods, just say “true” or “false”.",
                    "Bio Generation": "Tell me a bio about",
                    # "ASQA": "Instruction: Write an accurate, engaging, and concise answer for the given question using only the provided search results (some of which might be irrelevant) and cite them properly. Use an unbiased and journalistic tone. Always cite for any factual claim. When citing several search results, use [1][2][3]. Cite at least one document and at most three documents in each sentence. If multiple documents support the sentence, only cite a minimum sufficient subset of the documents.",
                    # first one is for baseline, and the second one is for self-rag.
                    "ASQA":  "Answer the following question. The question may be ambiguous and have multiple correct answers, and in that case, you have to provide a long-form answer including all correct answers."
                    }