from transformers import pipeline

summarizer = pipeline(task = "summarization"
                      , model = "t5-small"
                      , min_length = 20
                      , max_length = 150
                      , model_kwargs={"cache_dir": "../models"})

article_america = """America has changed dramatically during recent years. Not only has the number of graduates in traditional engineering disciplines such as mechanical, civil, electrical, chemical, and aeronautical engineering declined, but in most of the premier American universities engineering curricula now concentrate on and encourage largely the study of engineering science. As a result, there 
are declining offerings in engineering subjects dealing with infrastructure, the environment, and related issues, and greater concentration on high technology subjects, largely supporting increasingly complex scientific developments. While the latter is important, it should not be at the expense of more traditional engineering.
Rapidly developing economies such as China and India, as well as other industrial countries in Europe and Asia, continue to encourage and advance the teaching of engineering. Both China and India, respectively, graduate six and eight times as many traditional engineers as does the United States. 
Other industrial countries at minimum maintain their output, while America suffers an increasingly serious decline in the number of engineering graduates and a lack of well-educated engineers."""

article_india = """India is a land with a vast variety of wildlife and a large variety of cultures. Situated in South Asia’s heartland, India is a densely populated country. It is a vastly diverse country in terms of culture, climate, religion, and language. India has chosen a number of emblems to represent our country’s image. Saffron, white, and green make up the Indian national flag. The Ashok chakra in the centre has a navy blue 24-spoke wheel that represents virtue. 
India is well-known for possessing the world’s greatest cultural diversity. Even for Indians, visiting and exploring every culture in India is quite difficult. India’s various cultures attract visitors from all over the world who want to come here at least once in their lives to experience India’s rich diversity.
India is a secular and democratic country that gives the liberty to practise any religion. Along with that, every individual in India has the liberty to read any religious book of their choice. Every individual has the liberty to move to any part of the country and adapt to the culture of that region. Every state of India has its own official language.
Jana Gana Mana is our national anthem, while Vande Matram is our national song. In the ‘Lion Capital of Asoka’, India’s national emblem, four lions sit back to back on a cylindrical base with four Ashok chakras on each side, only one of which is visible in the front. There are three lions visible and one concealed. It is a sign of sovereignty that also represents strength and bravery. It is a beautiful country that excels in art, culture, architecture, education, etc."""

summarized_articles = summarizer([article_india, article_america])

for summarized_article in summarized_articles:
    print(summarized_article)
