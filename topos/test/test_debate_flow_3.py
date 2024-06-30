# test_debate_flow.py

import os
import unittest
import re

import warnings
from datetime import datetime
from topos.services.database.app_state import AppState, Neo4jConnection
from topos.FC.ontological_feature_detection import OntologicalFeatureDetection
from topos.channel.debatesim import DebateSimulator
from dotenv import load_dotenv
from uuid import uuid4
import json
from fastapi import FastAPI, WebSocket
from fastapi.testclient import TestClient
import asyncio


class TestDebateFlow(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        AppState._instance = None
        load_dotenv()  # Load environment variables

        self.neo4j_uri = os.getenv("NEO4J_URI")
        self.neo4j_user = os.getenv("NEO4J_USER")
        self.neo4j_password = os.getenv("NEO4J_PASSWORD")
        self.neo4j_test_database = os.getenv("NEO4J_TEST_DATABASE")

        # Initialize app state with Neo4j connection details
        self.app_state = AppState(self.neo4j_uri, self.neo4j_user, self.neo4j_password, self.neo4j_test_database)

        # Initialize the Neo4j connection
        self.neo4j_conn = Neo4jConnection(self.neo4j_uri, self.neo4j_user, self.neo4j_password)

        # Initialize the ontological feature detection with the test database
        self.ofd = OntologicalFeatureDetection(self.neo4j_uri, self.neo4j_user, self.neo4j_password,
                                               self.neo4j_test_database, use_neo4j=False)

        # Initialize DebateSimulator
        self.debate_simulator = DebateSimulator()

        # Clean the database before each test
        self.clean_database()

    async def asyncTearDown(self):
        # Close the connection properly

        # Get the existing instance of AppState
        app_state = AppState.get_instance()
        app_state.close()
        # Reset the singleton instance
        Neo4jConnection._instance = None

    def clean_database(self):
        app_state = AppState.get_instance()
        with app_state.driver.session(database=self.neo4j_test_database) as session:
            session.run("MATCH (n) DETACH DELETE n")

    def break_into_sentences(self, messages, min_words=20):
        output = []
        for message in messages:
            content = message["data"]["content"]
            sentences = re.split(r'[.!?]+', content)
            sentences = [s.strip() for s in sentences if s.strip()]

            current_sentence = []

            for sentence in sentences:
                words = sentence.split()
                if len(current_sentence) + len(words) >= min_words:
                    output.append({"role": message["role"], "data": {"user_id": message["data"]["user_id"],
                                                                     "content": " ".join(current_sentence)}})
                    current_sentence = words
                else:
                    current_sentence.extend(words)

            if current_sentence:
                output.append({"role": message["role"],
                               "data": {"user_id": message["data"]["user_id"], "content": " ".join(current_sentence)}})

        return output

    async def test_debate_flow(self):
        app = FastAPI()

        @app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            await websocket.accept()

            app_state = AppState.get_instance()

            app_state.set_state("user_id", f"user_{str(uuid4())}")
            app_state.set_state("session_id", f"session_{str(uuid4())}")
            app_state.set_state("prior_ontology", [])

            message_data = [
                {"role": "user", "data": {"user_id": "BIDEN",
                                          "content": "You have to take a look at what I was left when I became president, what Mr. Trump left me. We had an economy that was in freefall. The pandemic are so badly handled, many people were dying. All he said was, it's not that serious. Just inject a little bleach in your arm. It'd be all right. The economy collapsed. There were no jobs. Unemployment rate rose to 15 percent. It was terrible. And so, what we had to do is try to put things back together again. That's exactly what we began to do. We created 15,000 new jobs. We brought on – in a position where we have 800,000 new manufacturing jobs. But there's more to be done. There's more to be done. Working class people are still in trouble. I come from Scranton, Pennsylvania. I come from a household where the kitchen table – if things weren't able to be met during the month was a problem. Price of eggs, the price of gas, the price of housing, the price of a whole range of things. That's why I'm working so hard to make sure I deal with those problems. And we're going to make sure that we reduce the price of housing. We're going to make sure we build 2 million new units. We're going to make sure we cap rents, so corporate greed can't take over. The combination of what I was left and then corporate greed are the reason why we're in this problem right now. In addition to that, we're in a situation where if you had – take a look at all that was done in his administration, he didn't do much at all. By the time he left, there's – things had been in chaos. There was (ph) literally chaos. And so, we put things back together. We created, as I said, those (ph) jobs. We made sure we had a situation where we now – we brought down the price of prescription drugs, which is a major issue for many people, to $15 for – for an insulin shot, as opposed to $400. No senior has to pay more than $200 for any drug – all the drugs they (inaudible) beginning next year. And the situation is making – and we're going to make that available to everybody, to all Americans. So we're working to bring down the prices around the kitchen table. And that's what we're going to get done."}},
                {"role": "user", "data": {"user_id": "TRUMP",
                                          "content": "We had the greatest economy in the history of our country. We had never done so well. Every – everybody was amazed by it. Other countries were copying us. We got hit with COVID. And when we did, we spent the money necessary so we wouldn't end up in a Clutch Plague the likes of which we had in 1929. By the time we finished – so we did a great job. We got a lot of credit for the economy, a lot of credit for the military, and no wars and so many other things. Everything was rocking good. But the thing we never got the credit for, and we should have, is getting us out of that COVID mess. He created mandates; that was a disaster for our country. But other than that, we had – we had given them back a – a country where the stock market actually was higher than pre-COVID, and nobody thought that was even possible. The only jobs he created are for illegal immigrants and bounceback jobs; they're bounced back from the COVID. He has not done a good job. He's done a poor job. And inflation's killing our country. It is absolutely killing us."}},
                {"role": "user", "data": {"user_id": "BIDEN",
                                          "content": "Well, look, the greatest economy in the world, he's the only one who thinks that, I think. I don't know anybody else who thinks it was great – he had the greatest economy in the world. And, you know, the fact of the matter is that we found ourselves in a situation where his economy – he rewarded the wealthy. He had the largest tax cut in American history, $2 trillion. He raised the deficit larger than any president has in any one term. He's the only president other than Herbert Hoover who has lost more jobs than he had when he began, since Herbert Hoover. The idea that he did something that was significant. And the military – you know, when he was president, they were still killing people in Afghanistan. He didn't do anything about that. When he was president, we still found ourselves in a position where you had a notion that we were this safe country. The truth is, I'm the only president this century that doesn't have any – this – this decade – doesn't have any troops dying anywhere in the world, like he did."}},
                {"role": "user", "data": {"user_id": "TRUMP",
                                          "content": "Not going to drive them higher. It's just going to cause countries that have been ripping us off for years, like China and many others, in all fairness to China – it's going to just force them to pay us a lot of money, reduce our deficit tremendously, and give us a lot of power for other things. But he – he made a statement. The only thing he was right about is I gave you the largest tax cut in history. I also gave you the largest regulation cut in history. That's why we had all the jobs. And the jobs went down and then they bounced back and he's taking credit for bounceback jobs. You can't do that. He also said he inherited 9 percent inflation. No, he inherited almost no inflation and it stayed that way for 14 months. And then it blew up under his leadership, because they spent money like a bunch of people that didn't know what they were doing. And they don't know what they were doing. It was the worst – probably the worst administration in history. There's never been. And as far as Afghanistan is concerned, I was getting out of Afghanistan, but we were getting out with dignity, with strength, with power. He got out, it was the most embarrassing day in the history of our country's life."}},
                {"role": "user", "data": {"user_id": "TRUMP",
                                          "content": "Because the tax cuts spurred the greatest economy that we've ever seen just prior to COVID, and even after COVID. It was so strong that we were able to get through COVID much better than just about any other country. But we spurred – that tax spurred. Now, when we cut the taxes – as an example, the corporate tax was cut down to 21 percent from 39 percent, plus beyond that – we took in more revenue with much less tax and companies were bringing back trillions of dollars back into our country. The country was going like never before. And we were ready to start paying down debt. We were ready to start using the liquid gold right under our feet, the oil and gas right under our feet. We were going to have something that nobody else has had. We got hit with COVID. We did a lot to fix it. I gave him an unbelievable situation, with all of the therapeutics and all of the things that we came up with. We – we gave him something great. Remember, more people died under his administration, even though we had largely fixed it. More people died under his administration than our administration, and we were right in the middle of it. Something which a lot of people don't like to talk about, but he had far more people dying in his administration. He did the mandate, which is a disaster. Mandating it. The vaccine went out. He did a mandate on the vaccine, which is the thing that people most objected to about the vaccine. And he did a very poor job, just a very poor job. And I will tell you, not only poor there, but throughout the entire world, we're no longer respected as a country. They don't respect our leadership. They don't respect the United States anymore. We're like a Third World nation. Between weaponization of his election, trying to go after his political opponent, all of the things he's done, we've become like a Third World nation. And it's a shame the damage he's done to our country. And I'd love to ask him, and will, why he allowed millions of people to come in here from prisons, jails and mental institutions to come into our country and destroy our country."}},
                {"role": "user", "data": {"user_id": "BIDEN",
                                          "content": "He had the largest national debt of any president four-year period, number one. Number two, he got $2 trillion tax cut, benefited the very wealthy. What I'm going to do is fix the taxes. For example, we have a thousand trillionaires in America – I mean, billionaires in America. And what's happening? They're in a situation where they, in fact, pay 8.2 percent in taxes. If they just paid 24 percent or 25 percent, either one of those numbers, they'd raised $500 million – billion dollars, I should say, in a 10-year period. We'd be able to right – wipe out his debt. We'd be able to help make sure that – all those things we need to do, childcare, elder care, making sure that we continue to strengthen our healthcare system, making sure that we're able to make every single solitary person eligible for what I've been able to do with the COVID – excuse me, with dealing with everything we have to do with. Look, if – we finally beat Medicare."}},
                {"role": "user", "data": {"user_id": "TRUMP",
                                          "content": "Well, he's right: He did beat Medicaid (ph). He beat it to death. And he's destroying Medicare, because all of these people are coming in, they're putting them on Medicare, they're putting them on Social Security. They're going to destroy Social Security. This man is going to single-handedly destroy Social Security. These millions and millions of people coming in, they're trying to put them on Social Security. He will wipe out Social Security. He will wipe out Medicare. So he was right in the way he finished that sentence, and it's a shame. What's happened to our country in the last four years is not to be believed. Foreign countries – I'm friends with a lot of people. They cannot believe what happened to the United States of America. We're no longer respected. They don't like us. We give them everything they want, and they – they think we're stupid. They think we're very stupid people. What we're doing for other countries, and they do nothing for us. What this man has done to our country is absolutely criminal."}},
                {"role": "user", "data": {"user_id": "TRUMP",
                                          "content": "First of all, the Supreme Court just approved the abortion pill. And I agree with their decision to have done that, and I will not block it. And if you look at this whole question that you're asking, a complex, but not really complex – 51 years ago, you had Roe v. Wade, and everybody wanted to get it back to the states, everybody, without exception. Democrats, Republicans, liberals, conservatives, everybody wanted it back. Religious leaders. And what I did is I put three great Supreme Court justices on the court, and they happened to vote in favor of killing Roe v. Wade and moving it back to the states. This is something that everybody wanted. Now, 10 years ago or so, they started talking about how many weeks and how many of this – getting into other things, But every legal scholar, throughout the world, the most respected, wanted it brought back to the states. I did that. Now the states are working it out. If you look at Ohio, it was a decision that was – that was an end result that was a little bit more liberal than you would have thought. Kansas I would say the same thing. Texas is different. Florida is different. But they're all making their own decisions right now. And right now, the states control it. That's the vote of the people. Like Ronald Reagan, I believe in the exceptions. I am a person that believes. And frankly, I think it's important to believe in the exceptions. Some people – you have to follow your heart. Some people don't believe in that. But I believe in the exceptions for rape, incest and the life of the mother. I think it's very important. Some people don't. Follow your heart. But you have to get elected also and – because that has to do with other things. You got to get elected. The problem they have is they're radical, because they will take the life of a child in the eighth month, the ninth month, and even after birth – after birth. If you look at the former governor of Virginia, he was willing to do this. He said, we'll put the baby aside and we'll determine what we do with the baby. Meaning, we'll kill the baby. What happened is we brought it back to the states and the country is now coming together on this issue. It's been a great thing."}},
                {"role": "user", "data": {"user_id": "BIDEN",
                                          "content": "It's been a terrible thing what you've done. The fact is that the vast majority of constitutional scholars supported Roe when it was decided, supported Roe. And I was – that's – this idea that they were all against it is just ridiculous.And this is the guy who says the states should be able to have it. We're in a state where in six weeks you don't even know whether you're pregnant or not, but you cannot see a doctor, have your – and have him decide on what your circumstances are, whether you need help. The idea that states are able to do this is a little like saying, we're going to turn civil rights back to the states, let each state have a different rule. Look, there's so many young women who have been – including a young woman who just was murdered and he went to the funeral. The idea that she was murdered by – by – by an immigrant coming in and (inaudible) talk about that. But here's the deal, there's a lot of young women who are being raped by their – by their in-laws, by their – by their spouses, brothers and sisters, by – just – it's just – it's just ridiculous. And they can do nothing about it. And they try to arrest them when they cross state lines."}},
                {"role": "user", "data": {"user_id": "TRUMP",
                                          "content": "There have been many young women murdered by the same people he allows to come across our border. We have a border that's the most dangerous place anywhere in the world – considered the most dangerous place anywhere in the world. And he opened it up, and these killers are coming into our country, and they are raping and killing women. And it's a terrible thing. As far as the abortion's concerned, it is now back with the states. The states are voting and in many cases, they – it's, frankly, a very liberal decision. In many cases, it's the opposite. But they're voting and it's bringing it back to the vote of the people, which is what everybody wanted, including the founders, if they knew about this issue, which frankly they didn't, but they would have – everybody want it brought back. Ronald Reagan wanted it brought back. He wasn't able to get it. Everybody wanted it brought back and many presidents had tried to get it back. I was the one to do it. And again, this gives it the vote of the people. And that's where they wanted it. Every legal scholar wanted it that way."}},
                {"role": "user", "data": {"user_id": "BIDEN",
                                          "content": "I supported Roe v. Wade, which had three trimesters. First time is between a woman and a doctor. Second time is between a doctor and an extreme situation. A third time is between the doctor – I mean, it'd be between the woman and the state. The idea that the politicians – that the founders wanted the politicians to be the ones making decisions about a woman's health is ridiculous. That's the last – no politician should be making that decision. A doctor should be making those decisions. That's how it should be run. That's what you're going to do. And if I'm elected, I'm going to restore Roe v. Wade."}}
            ]

            message_data = self.break_into_sentences(message_data)

            expected_messages_count = {
                "BIDEN": 0,
                "TRUMP": 0
            }

            built_message_history = []

            # send messages one by one to ensure the graph database gets filled out step by step
            for message in message_data:
                data = json.dumps({
                    "user_id": message["data"]["user_id"],
                    "message": message["data"]["content"],
                    "message_history": built_message_history,
                    "model": "dolphin-llama3",
                    "temperature": 0.3,
                    "topic": "Climate Change: Is human activity the primary cause?"
                })
                await self.debate_simulator.debate_step(websocket, data, app_state)

                built_message_history.append(message)

                # Update expected messages count
                user_id = message["data"]["user_id"]
                expected_messages_count[user_id] += 1

                # Print messages for each user after each step
                user_messages = self.ofd.get_messages_by_user(user_id, "SENT")
                print("\n\n\n")
                print(f"Messages for user {user_id}:")
                for msg in user_messages:
                    print(f"Message ID: {msg['message_id']}, Content: {msg['message']}, Timestamp: {msg['timestamp']}")

                # Assert the number of messages retrieved
                self.assertEqual(len(user_messages), expected_messages_count[user_id])
                print("\n\n\n")

            # Final message to get response
            data = json.dumps({
                "user_id": "userA",
                "message": "Mitigating climate change requires a global effort to reduce emissions and transition to renewable energy sources.",
                "message_history": message_data,
                "model": "dolphin-llama3",
                "temperature": 0.3,
                "topic": "Climate Change: Is human activity the primary cause?"
            })
            await self.debate_simulator.debate_step(websocket, data, app_state)

            kl_divergences = app_state.get_value("kl_divergences", [])
            print(f"KL-Divergences: {kl_divergences}")

        client = TestClient(app)

        with client.websocket_connect("/ws") as websocket:
            completed = False
            full_response = ""
            while not completed:
                response = websocket.receive_json()
                if response["status"] == "generating":
                    full_response += response["response"]
                elif response["status"] == "completed":
                    full_response += response["response"]
                    completed = True
                    self.assertIn("semantic_category", response)
                    self.assertTrue(response["completed"])

        if __name__ == "__main__":
            unittest.main()
