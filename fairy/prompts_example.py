def orca_format(text):
    system_message = "You are Orca, an AI language model created by Microsoft. You are a helpful assistant. Please give me a list of atomic facts of the following texts."

    prompt = """<|im_start|>system
{}<|im_end|>
<|im_start|>user
lisa courtney, of hertfordshire, has spent most of her life collecting pokemon memorabilia.<|im_end|>
<|im_start|>assistant
- Lisa Courtney is from Hertfordshire.
- Lisa Courtney has spent most of her life collecting Pokémon memorabilia.<|im_end|>
<|im_start|>user
prince jan zylinski said he was fed up with discrimination against poles living in britain.<|im_end|>
<|im_start|>assistant
- Prince Jan Zylinski made a statement.
- The statement made by Prince Jan Zylinski was about discrimination.
- The statement made by Prince Jan Zylinski was regarding Poles living in Britain.
- Prince Jan Zylinski expressed feeling fed up with this type of discrimination.<|im_end|>
<|im_start|>user
no charges were filed, there will be no travel ban.<|im_end|>
<|im_start|>assistant
- No charges were filed.
- There will be no travel ban.<|im_end|>
<|im_start|>user
rudd has pleaded guilty to threatening to kill and possession of drugs in a court.<|im_end|>
<|im_start|>assistant
- Rudd has pleaded guilty.
- Rudd has pleaded guilty to threatening to kill.
- Rudd has pleaded guilty to possession of drugs.<|im_end|>
<|im_start|>user
Lee made his acting debut in the film The Moon is the Sun's Dream (1992), and continued to appear in small and supporting roles throughout the 1990s.<|im_end|>
<|im_start|>assistant
- Lee made his acting debut in The Moon is the Sun's Dream.
- The Moon is the Sun’s Dream is a film.
- The Moon is the Sun’s Dream was released in 1992.
- After Lee's acting debut, he appeared in small and supporting roles throughout the 1990s.<|im_end|>
<|im_start|>user
In 1963, Collins became one of the third group of astronauts selected by NASA and he served as the back-up Command Module Pilot for the Gemini 7 mission.<|im_end|>
<|im_start|>assistant
- Collins became an astronaut.
- Collins became one of the third group of astronauts selected by NASA in 1963.
- Collins served as the Command Module Pilot for the Gemini 7 mission.<|im_end|>
<|im_start|>user
In addition to his acting roles, Bateman has written and directed two short films and is currently in development on his feature debut.<|im_end|>
<|im_start|>assistant
- Bateman has acting roles.
- Bateman has written two short films.
- Bateman has directed two short films.
- Bateman is currently in development on his feature debut.<|im_end|>
<|im_start|>user
Michael Collins (born October 31, 1930) is a retired American astronaut and test pilot who was the Command Module Pilot for the Apollo 11 mission in 1969.<|im_end|>
<|im_start|>assistant
- Michael Collins was born on October 31, 1930.
- Michael Collins is retired.
- Michael Collins is an American.
- Michael Collins was an astronaut.
- Michael Collins was a test pilot.
- Michael Collins was the Command Module Pilot for the Apollo 11 mission in 1969.<|im_end|>
<|im_start|>user
{}<|im_end|>
<|im_start|>assistant
""".format(system_message, text)
    return prompt

def zephyr_format(text):
    messages = [
{
    "role": "system",
    "content": "You are a helpful assistant. Please give me a list of atomic facts of the following texts."
},
{"role": "user", "content": "lisa courtney, of hertfordshire, has spent most of her life collecting pokemon memorabilia."},
{"role": "assistant", "content": """
- Lisa Courtney is from Hertfordshire.
- Lisa Courtney has spent most of her life collecting Pokémon memorabilia."""},
{"role": "user", "content": "prince jan zylinski said he was fed up with discrimination against poles living in britain."},
{"role": "assistant", "content": """
- Prince Jan Zylinski made a statement.
- The statement made by Prince Jan Zylinski was about discrimination.
- The statement made by Prince Jan Zylinski was regarding Poles living in Britain.
- Prince Jan Zylinski expressed feeling fed up with this type of discrimination."""},
{"role": "user", "content": "no charges were filed, there will be no travel ban."},
{"role": "assistant", "content": """
- No charges were filed.
- There will be no travel ban."""},
{"role": "user", "content": "rudd has pleaded guilty to threatening to kill and possession of drugs in a court."},
{"role": "assistant", "content": """
- Rudd has pleaded guilty.
- Rudd has pleaded guilty to threatening to kill.
- Rudd has pleaded guilty to possession of drugs."""},
{"role": "user", "content": "Lee made his acting debut in the film The Moon is the Sun's Dream (1992), and continued to appear in small and supporting roles throughout the 1990s."},
{"role": "assistant", "content": """
- Lee made his acting debut in The Moon is the Sun's Dream.
- The Moon is the Sun’s Dream is a film.
- The Moon is the Sun’s Dream was released in 1992.
- After Lee's acting debut, he appeared in small and supporting roles throughout the 1990s."""},
{"role": "user", "content": "In 1963, Collins became one of the third group of astronauts selected by NASA and he served as the back-up Command Module Pilot for the Gemini 7 mission."},
{"role": "assistant", "content": """
- Collins became an astronaut.
- Collins became one of the third group of astronauts selected by NASA in 1963.
- Collins served as the Command Module Pilot for the Gemini 7 mission."""},
{"role": "user", "content": "In addition to his acting roles, Bateman has written and directed two short films and is currently in development on his feature debut."},
{"role": "assistant", "content": """
- Bateman has acting roles.
- Bateman has written two short films.
- Bateman has directed two short films.
- Bateman is currently in development on his feature debut."""},
{"role": "user", "content": "Michael Collins (born October 31, 1930) is a retired American astronaut and test pilot who was the Command Module Pilot for the Apollo 11 mission in 1969."},
{"role": "assistant", "content": """
- Michael Collins was born on October 31, 1930.
- Michael Collins is retired.
- Michael Collins is an American.
- Michael Collins was an astronaut.
- Michael Collins was a test pilot.
- Michael Collins was the Command Module Pilot for the Apollo 11 mission in 1969."""},
{"role": "user", "content": "{}\n".format(text)}]
    return messages

def mistral_format(text):
    messages = [
        {"role": "user", "content": "Please give me a list of atomic facts of the following texts: lisa courtney, of hertfordshire, has spent most of her life collecting pokemon memorabilia."},
        {"role": "assistant", "content": """
- Lisa Courtney is from Hertfordshire.
- Lisa Courtney has spent most of her life collecting Pokémon memorabilia."""},
        {"role": "user", "content": "Please give me a list of atomic facts of the following texts: prince jan zylinski said he was fed up with discrimination against poles living in britain."},
        {"role": "assistant", "content": """
- Prince Jan Zylinski made a statement.
- The statement made by Prince Jan Zylinski was about discrimination.
- The statement made by Prince Jan Zylinski was regarding Poles living in Britain.
- Prince Jan Zylinski expressed feeling fed up with this type of discrimination."""},
        {"role": "user", "content": "Please give me a list of atomic facts of the following texts: no charges were filed, there will be no travel ban."},
        {"role": "assistant", "content": """
- No charges were filed.
- There will be no travel ban."""},
        {"role": "user", "content": "Please give me a list of atomic facts of the following texts: rudd has pleaded guilty to threatening to kill and possession of drugs in a court."},
        {"role": "assistant", "content": """
- Rudd has pleaded guilty.
- Rudd has pleaded guilty to threatening to kill.
- Rudd has pleaded guilty to possession of drugs."""},
        {"role": "user", "content": "Please give me a list of atomic facts of the following texts: Lee made his acting debut in the film The Moon is the Sun's Dream (1992), and continued to appear in small and supporting roles throughout the 1990s."},
        {"role": "assistant", "content": """
- Lee made his acting debut in The Moon is the Sun's Dream.
- The Moon is the Sun’s Dream is a film.
- The Moon is the Sun’s Dream was released in 1992.
- After Lee's acting debut, he appeared in small and supporting roles throughout the 1990s."""},
        {"role": "user", "content": "Please give me a list of atomic facts of the following texts: In 1963, Collins became one of the third group of astronauts selected by NASA and he served as the back-up Command Module Pilot for the Gemini 7 mission."},
        {"role": "assistant", "content": """
- Collins became an astronaut.
- Collins became one of the third group of astronauts selected by NASA in 1963.
- Collins served as the Command Module Pilot for the Gemini 7 mission."""},
        {"role": "user", "content": "Please give me a list of atomic facts of the following texts: In addition to his acting roles, Bateman has written and directed two short films and is currently in development on his feature debut."},
        {"role": "assistant", "content": """
- Bateman has acting roles.
- Bateman has written two short films.
- Bateman has directed two short films.
- Bateman is currently in development on his feature debut."""},
        {"role": "user", "content": "Please give me a list of atomic facts of the following texts: Michael Collins (born October 31, 1930) is a retired American astronaut and test pilot who was the Command Module Pilot for the Apollo 11 mission in 1969."},
        {"role": "assistant", "content": """
- Michael Collins was born on October 31, 1930.
- Michael Collins is retired.
- Michael Collins is an American.
- Michael Collins was an astronaut.
- Michael Collins was a test pilot.
- Michael Collins was the Command Module Pilot for the Apollo 11 mission in 1969."""},
        {"role": "user", "content": "Please give me a list of atomic facts of the following texts: {}".format(text)}]
    return messages