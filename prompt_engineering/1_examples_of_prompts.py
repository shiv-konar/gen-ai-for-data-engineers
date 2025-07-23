from transformers import pipeline

pipe = pipeline(task="text2text-generation", model="google/flan-t5-base", model_kwargs={"cache_dir": "../models"})

input_text = "I'm very happy to see you"
translation_prompt = "Translate from English to French: " + input_text
result = pipe(translation_prompt)
print(result)

# tokenizer = pipe.tokenizer
# device = torch.device("cpu")
# model = pipe.model.to(device)
# inputs = tokenizer(prompt, return_tensors="pt")
# outputs = model.generate(**inputs)
# print(tokenizer.decode(outputs[0], skip_special_characters=True))

article = """India is a land with a vast variety of wildlife and a large variety of cultures. Situated in South Asia’s heartland, India is a densely populated country. It is a vastly diverse country in terms of culture, climate, religion, and language. India has chosen a number of emblems to represent our country’s image. Saffron, white, and green make up the Indian national flag. The Ashok chakra in the centre has a navy blue 24-spoke wheel that represents virtue. 
India is well-known for possessing the world’s greatest cultural diversity. Even for Indians, visiting and exploring every culture in India is quite difficult. India’s various cultures attract visitors from all over the world who want to come here at least once in their lives to experience India’s rich diversity.
India is a secular and democratic country that gives the liberty to practise any religion. Along with that, every individual in India has the liberty to read any religious book of their choice. Every individual has the liberty to move to any part of the country and adapt to the culture of that region. Every state of India has its own official language.
Jana Gana Mana is our national anthem, while Vande Matram is our national song. In the ‘Lion Capital of Asoka’, India’s national emblem, four lions sit back to back on a cylindrical base with four Ashok chakras on each side, only one of which is visible in the front. There are three lions visible and one concealed. It is a sign of sovereignty that also represents strength and bravery. It is a beautiful country that excels in art, culture, architecture, education, etc."""

summarization_prompt = "Summarize: " + article

result = pipe(summarization_prompt, min_length=100, max_length=200)
print(result)

tweet = """This movie is definitely one of my favorite movies of its kind. 
The interaction between respectable and morally strong characters is an ode 
to chivalry and the honor code amongst thieves and policemen."""

sentiment_prompt = "Classify the given tweet into neutral, positive or negative: " + tweet
result = pipe(sentiment_prompt)
print(result)

context = """Gazpacho is a cold soup and drink made of raw, blended vegetables. Most gazpacho includes stale bread, tomato, cucumbers, onion, bell peppers, garlic, olive oil, wine vinegar, water, and salt. Northern recipes often include cumin and/or pimentón (smoked sweet paprika). Traditionally, gazpacho was made by pounding the vegetables in a mortar with a pestle; this more laborious method is still sometimes used as it helps keep the gazpacho cool and avoids the foam and silky consistency of smoothie versions made in blenders or food processors."""
question = "What modern tool is used to make gazpacho?"

qa_prompt = "Answer the question using the given context. \n" + "Context: " + context + "\n" + "Question: " + question
result = pipe(qa_prompt)
print(result)

text = "The Golden State Warriors are an American professional basketball team based in San Francisco"
ner_prompt = "Return a comma separated list of named entities in the text.\n" +  "Text: " + text
result = pipe(ner_prompt)
print(result)

