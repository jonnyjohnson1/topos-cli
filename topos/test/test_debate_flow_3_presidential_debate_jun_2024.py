# test_debate_flow_3_presidential_debate_jun_2024.py

import os
import unittest
from uuid import uuid4
from datetime import datetime, timedelta, UTC
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, status
from fastapi.testclient import TestClient
from fastapi.security import OAuth2PasswordBearer
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
import json
import jwt
from jwt.exceptions import InvalidTokenError
import asyncio
import threading

from topos.services.database.app_state import AppState
from topos.channel.debatesim import DebateSimulator
from topos.api.debate_routes import router, SECRET_KEY, ALGORITHM

app = FastAPI()
app.include_router(router)


class WebSocketThread(threading.Thread):
    def __init__(self, url, messages, responses):
        threading.Thread.__init__(self)
        self.url = url
        self.messages = messages
        self.responses = responses
        self.client = TestClient(app)

    def run(self):
        with self.client.websocket_connect(self.url) as websocket:
            for message in self.messages:
                websocket.send_json(message)
                response = websocket.receive_json()
                self.responses.append(response)

class TestDebateJWTFlow(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        AppState._instance = None
        load_dotenv()  # Load environment variables

        # Initialize app state
        self.app_state = AppState(use_neo4j=False)

        # Initialize DebateSimulator without Neo4j
        self.debate_simulator = await DebateSimulator.get_instance()

    async def asyncTearDown(self):
        # Make sure to cancel the processing task when tearing down
        if self.debate_simulator.processing_task:
            self.debate_simulator.processing_task.cancel()
            try:
                await self.debate_simulator.processing_task
            except asyncio.CancelledError:
                pass

        # Reset the singleton instance
        AppState._instance = None

    def test_jwt_generation_and_validation(self):
        user_id = f"user_{str(uuid4())}"
        session_id = f"session_{str(uuid4())}"

        # Generate JWT token
        token_data = {
            "user_id": user_id,
            "exp": datetime.now(UTC) + timedelta(hours=1)
        }
        token = jwt.encode(token_data, SECRET_KEY, algorithm=ALGORITHM)
        self.assertIsNotNone(token)

        # Validate JWT token
        try:
            decoded_token = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
            self.assertEqual(decoded_token["user_id"], user_id)
        except InvalidTokenError:
            self.fail("JWT token validation failed")

    def test_debate_flow_with_jwt(self):
        client = TestClient(app)

        response = client.post("/admin_set_accounts", data={"BIDEN": "pass", "TRUMP": "pass"})
        self.assertEqual(response.status_code, 200)

        # Create tokens for two users
        response = client.post("/token", data={"username": "BIDEN", "password": "pass"})
        self.assertEqual(response.status_code, 200)
        token_user_a = response.json().get("access_token")
        self.assertIsNotNone(token_user_a)

        response = client.post("/token", data={"username": "TRUMP", "password": "pass"})
        self.assertEqual(response.status_code, 200)
        token_user_b = response.json().get("access_token")
        self.assertIsNotNone(token_user_b)

        # Get or create session for userA
        response = client.get("/sessions", headers={"Authorization": f"Bearer {token_user_a}"})
        self.assertEqual(response.status_code, 200)
        sessions = response.json().get("sessions", [])

        if not sessions:
            response = client.post("/create_session", headers={"Authorization": f"Bearer {token_user_a}"})
            self.assertEqual(response.status_code, 200)
            session_id = response.json().get("session_id")
            self.assertIsNotNone(session_id)
        else:
            session_id = sessions[0]

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

        unique_users = set(message["data"]["user_id"] for message in message_data)
        user_a_name, user_b_name = list(unique_users)

        message_data = self.debate_simulator.break_into_sentences(message_data)

        # Open WebSocket connections for both users
        with client.websocket_connect(f"/ws?token={token_user_a}&session_id={session_id}") as websocket_a, \
             client.websocket_connect(f"/ws?token={token_user_b}&session_id={session_id}") as websocket_b:

            for message in message_data:
                if message["data"]["user_id"] == user_a_name:
                    websocket_a.send_json({
                        "message": message["data"]["content"],
                        "user_id": message["data"]["user_id"],
                        "generation_nonce": str(uuid4())
                    })
                else:
                    websocket_b.send_json({
                        "message": message["data"]["content"],
                        "user_id": message["data"]["user_id"],
                        "generation_nonce": str(uuid4())
                    })

                print(f"\t[ Sent message: {message['data']['content']} ]")

                initial_response_received = False
                clusters_received = False
                updated_clusters_received = False
                wepcc_result_received = False
                final_results_received = False

                # Wait for and process multiple responses
                while not (initial_response_received and clusters_received and updated_clusters_received
                           and wepcc_result_received and final_results_received):
                    if message["data"]["user_id"] == user_a_name:
                        response = websocket_a.receive_json()
                    else:
                        response = websocket_b.receive_json()

                    # print(f"\t\t[ Received response: {response} ]")

                    if response["status"] == "message_processed":
                        self.assertIn("initial_analysis", response)
                        initial_response_received = True

                    if response["status"] == "initial_clusters":
                        self.assertIn("clusters", response)
                        clusters_received = True

                    if response["status"] == "updated_clusters":
                        self.assertIn("clusters", response)
                        updated_clusters_received = True

                    if response["status"] == "wepcc_result":
                        self.assertIn("wepcc_result", response)
                        wepcc_result_received = True

                    if response["status"] == "final_results":
                        self.assertIn("results", response)
                        final_results_received = True

                print(f"\t[ Messaged processed: {message['data']['content']} ]")

        print("Test completed")

        # Verify that all expected responses were received
        self.assertTrue(initial_response_received, "Did not receive initial response.")
        self.assertTrue(clusters_received, "Did not receive initial clusters.")
        self.assertTrue(updated_clusters_received, "Did not receive updated clusters.")
        self.assertTrue(wepcc_result_received, "Did not receive WEPCC result.")
        self.assertTrue(final_results_received, "Did not receive final results.")


if __name__ == "__main__":

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    unittest.main()

