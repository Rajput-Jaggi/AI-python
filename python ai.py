import random
import json
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class MealPlanChatbot:
    def __init__(self):
        self.recipes = self._load_recipes()
        self.vectorizer = TfidfVectorizer()
        self._train_vectorizer()
        self.user_preferences = {}
        self.conversation_state = "welcome"
        self.meal_plan = None
        self.current_preference = None
        
    def _load_recipes(self):
        """Load recipe database from JSON file"""
        try:
            with open('recipes.json', 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            # Default recipes if file not found
            return [
                {"name": "Vegetable Stir Fry", "ingredients": ["bell pepper", "broccoli", "carrot", "soy sauce"], 
                 "diet": ["vegetarian", "vegan"], "meal_type": ["lunch", "dinner"], "calories": 320, "protein": 12, "carbs": 45, "fat": 8},
                {"name": "Greek Yogurt Parfait", "ingredients": ["greek yogurt", "granola", "berries", "honey"], 
                 "diet": ["vegetarian"], "meal_type": ["breakfast"], "calories": 250, "protein": 18, "carbs": 30, "fat": 6},
                {"name": "Quinoa Bowl", "ingredients": ["quinoa", "avocado", "black beans", "corn"], 
                 "diet": ["vegetarian", "vegan", "gluten-free"], "meal_type": ["lunch", "dinner"], "calories": 400, "protein": 15, "carbs": 55, "fat": 14},
                {"name": "Avocado Toast", "ingredients": ["bread", "avocado", "tomato", "salt"], 
                 "diet": ["vegetarian"], "meal_type": ["breakfast", "lunch"], "calories": 350, "protein": 10, "carbs": 40, "fat": 18}
            ]
    
    def _train_vectorizer(self):
        """Train TF-IDF vectorizer on recipe ingredients"""
        all_ingredients = [" ".join(recipe["ingredients"]) for recipe in self.recipes]
        self.vectorizer.fit(all_ingredients)
        
    def start_chat(self):
        """Start the interactive chatbot session"""
        print("\n🍎 Welcome to NutriBot - Your AI Meal Planning Assistant! 🥗")
        print("Type 'quit' at any time to exit.\n")
        print("Bot:", self.get_response("hello"))
        
        while True:
            user_input = input("You: ").strip().lower()
            
            if user_input == 'quit':
                print("Bot: Goodbye! Happy meal planning!")
                break
                
            response = self.get_response(user_input)
            print("Bot:", response)
            
            if self.conversation_state == "meal_plan_generated":
                self._handle_meal_plan_interaction(user_input)
    
    def get_response(self, user_input):
        """Generate appropriate response based on user input and conversation state"""
        if self.conversation_state == "welcome":
            return self._handle_welcome(user_input)
        elif self.conversation_state == "collecting_preferences":
            return self._handle_preferences(user_input)
        elif self.conversation_state == "confirm_plan":
            return self._handle_confirmation(user_input)
        else:
            return "I'm not sure how to respond to that. Can you rephrase?"
    
    def _handle_welcome(self, user_input):
        """Handle initial conversation"""
        responses = {
            'hello': "Hi there! I can help you create a personalized meal plan. Do you have any dietary restrictions? (e.g., vegetarian, vegan, gluten-free)",
            'hi': "Hello! Ready to plan some delicious meals? Do you have any food allergies or dietary preferences?",
            'hey': "Hey! Let's create your perfect meal plan. Any dietary restrictions I should know about?"
        }
        
        if user_input in responses:
            self.conversation_state = "collecting_preferences"
            return responses[user_input]
        else:
            self.conversation_state = "collecting_preferences"
            return "Great! Let's get started with your meal plan. First, do you have any dietary restrictions?"
    
    def _handle_preferences(self, user_input):
        """Collect user preferences"""
        if self.current_preference is None:
            self.current_preference = "dietary_restrictions"
            self.user_preferences = {"dietary_restrictions": []}
        
        if self.current_preference == "dietary_restrictions":
            # Extract dietary restrictions
            restrictions = []
            if 'none' in user_input:
                restrictions = []
            else:
                for diet in ["vegetarian", "vegan", "gluten-free", "dairy-free", "nut-free", "pescatarian"]:
                    if diet in user_input:
                        restrictions.append(diet)
            
            self.user_preferences["dietary_restrictions"] = restrictions
            self.current_preference = "calorie_goal"
            return "Got it! What's your daily calorie goal? (e.g., '2000 calories' or '1500-1800')"
        
        elif self.current_preference == "calorie_goal":
            # Parse calorie range
            match = re.search(r'(\d+)\s*-\s*(\d+)', user_input)
            if match:
                self.user_preferences["calorie_range"] = [int(match.group(1)), int(match.group(2))]
            else:
                match = re.search(r'(\d+)', user_input)
                if match:
                    cal = int(match.group(1))
                    self.user_preferences["calorie_range"] = [cal-200, cal+200]
                else:
                    self.user_preferences["calorie_range"] = [1600, 2200]
            
            self.current_preference = "ingredient_preferences"
            return "Thanks! Any favorite ingredients you'd like to include? (e.g., 'chicken, quinoa, avocado')"
        
        elif self.current_preference == "ingredient_preferences":
            # Extract ingredients
            if 'none' in user_input or 'no' in user_input:
                self.user_preferences["preferred_ingredients"] = []
            else:
                self.user_preferences["preferred_ingredients"] = [ing.strip() for ing in re.split(',|and', user_input)]
            
            self.current_preference = None
            self.conversation_state = "confirm_plan"
            return ("Great! Based on your preferences:\n"
                   f"- Dietary restrictions: {', '.join(self.user_preferences.get('dietary_restrictions', ['none']))}\n"
                   f"- Calorie range: {self.user_preferences.get('calorie_range', [1600, 2200])}\n"
                   f"- Preferred ingredients: {', '.join(self.user_preferences.get('preferred_ingredients', ['none']))}\n\n"
                   "Shall I generate your meal plan now? (yes/no)")
    
    def _handle_confirmation(self, user_input):
        """Handle user confirmation to generate meal plan"""
        if user_input.startswith('y'):
            self.meal_plan = self.generate_meal_plan(self.user_preferences)
            self.conversation_state = "meal_plan_generated"
            
            plan_str = "Here's your personalized meal plan:\n\n"
            for day in self.meal_plan["meal_plan"]:
                plan_str += f"Day {day['day']}:\n"
                for meal, recipe in day["meals"].items():
                    if recipe:
                        plan_str += f"- {meal.capitalize()}: {recipe['name']} ({recipe['calories']} cal)\n"
                plan_str += f"- Snack: {day['snack'] or 'None suggested'}\n\n"
            
            plan_str += ("Nutrition Totals:\n"
                       f"- Calories: {self.meal_plan['total_nutrition']['calories']}\n"
                       f"- Protein: {self.meal_plan['total_nutrition']['protein']}g\n"
                       f"- Carbs: {self.meal_plan['total_nutrition']['carbs']}g\n"
                       f"- Fat: {self.meal_plan['total_nutrition']['fat']}g\n\n"
                       "You can:\n"
                       "- Type 'shopping list' to see ingredients needed\n"
                       "- Type 'modify' to change your preferences\n"
                       "- Type 'new' to start over\n"
                       "- Type 'quit' to exit")
            
            return plan_str
        else:
            self.conversation_state = "collecting_preferences"
            self.current_preference = "dietary_restrictions"
            return "Okay, let's adjust your preferences. Do you have any dietary restrictions?"
    
    def _handle_meal_plan_interaction(self, user_input):
        """Handle user interactions after meal plan is generated"""
        if user_input == 'quit':
            print("Bot: Goodbye! Happy meal planning!")
            exit()
        elif 'shopping' in user_input:
            shopping_list = self.meal_plan["shopping_list"]["ingredients"]
            return "Here's your shopping list:\n" + "\n".join(f"- {ing} ({qty}x)" for ing, qty in shopping_list.items())
        elif 'modify' in user_input:
            self.conversation_state = "collecting_preferences"
            self.current_preference = "dietary_restrictions"
            return "Let's modify your preferences. Do you have any dietary restrictions?"
        elif 'new' in user_input:
            self.conversation_state = "welcome"
            self.user_preferences = {}
            return self.get_response("hello")
        else:
            return "I'm not sure what you mean. You can ask for 'shopping list', 'modify', 'new', or 'quit'"
    
    def generate_meal_plan(self, user_preferences, days=3):
        """
        Generate a meal plan based on user preferences
        """
        filtered_recipes = self._filter_recipes(user_preferences)
        
        if not filtered_recipes:
            return {"error": "No recipes match your preferences"}
            
        meal_plan = []
        total_nutrition = {"calories": 0, "protein": 0, "carbs": 0, "fat": 0}
        
        for day in range(1, days+1):
            meals = {
                "breakfast": self._select_meal(filtered_recipes, "breakfast", user_preferences),
                "lunch": self._select_meal(filtered_recipes, "lunch", user_preferences),
                "dinner": self._select_meal(filtered_recipes, "dinner", user_preferences)
            }
            
            # Update nutrition totals
            for meal in meals.values():
                if meal:
                    total_nutrition["calories"] += meal["calories"]
                    total_nutrition["protein"] += meal["protein"]
                    total_nutrition["carbs"] += meal["carbs"]
                    total_nutrition["fat"] += meal["fat"]
            
            meal_plan.append({
                "day": day,
                "meals": meals,
                "snack": self._suggest_snack(total_nutrition, user_preferences)
            })
        
        return {
            "meal_plan": meal_plan,
            "total_nutrition": total_nutrition,
            "shopping_list": self._generate_shopping_list(meal_plan)
        }
    
    def _filter_recipes(self, preferences):
        """
        Filter recipes based on user preferences
        """
        filtered = []
        
        for recipe in self.recipes:
            # Check dietary restrictions
            if preferences.get("dietary_restrictions"):
                diet_ok = any(diet in recipe["diet"] for diet in preferences["dietary_restrictions"])
                if not diet_ok:
                    continue
                    
            # Check calorie range
            if preferences.get("calorie_range"):
                if not (preferences["calorie_range"][0] <= recipe["calories"] <= preferences["calorie_range"][1]):
                    continue
                    
            # Check preferred ingredients if specified
            if preferences.get("preferred_ingredients"):
                ingredient_match = any(ing in recipe["ingredients"] for ing in preferences["preferred_ingredients"])
                if not ingredient_match and preferences.get("strict_ingredients", False):
                    continue
                    
            filtered.append(recipe)
            
        return filtered
    
    def _select_meal(self, recipes, meal_type, preferences):
        """
        Select an appropriate meal based on type (breakfast, lunch, dinner)
        """
        # Filter by meal type first
        suitable = [r for r in recipes if meal_type in r.get("meal_type", [])]
        
        if not suitable:
            return None
            
        # If user has ingredient preferences, prioritize those
        if preferences.get("preferred_ingredients"):
            # Calculate similarity between recipe ingredients and preferred ingredients
            pref_text = " ".join(preferences["preferred_ingredients"])
            recipe_texts = [" ".join(r["ingredients"]) for r in suitable]
            
            vectors = self.vectorizer.transform([pref_text] + recipe_texts)
            similarities = cosine_similarity(vectors[0:1], vectors[1:]).flatten()
            
            # Select top 3 most similar
            top_indices = np.argsort(similarities)[-3:][::-1]
            selected = random.choice([suitable[i] for i in top_indices])
        else:
            selected = random.choice(suitable)
            
        return selected
    
    def _suggest_snack(self, current_nutrition, preferences):
        """
        Suggest a snack based on current nutrition totals
        """
        target_calories = preferences.get("daily_calories", 2000)
        remaining_calories = max(0, target_calories - current_nutrition["calories"])
        
        if remaining_calories < 150:
            return None
            
        protein_ratio = current_nutrition["protein"] / current_nutrition["calories"] if current_nutrition["calories"] > 0 else 0
        carb_ratio = current_nutrition["carbs"] / current_nutrition["calories"] if current_nutrition["calories"] > 0 else 0
        
        if protein_ratio < 0.15:
            return "Protein shake or handful of nuts"
        elif carb_ratio < 0.45:
            return "Fruit or whole grain crackers"
        else:
            return "Vegetables with hummus"
    
    def _generate_shopping_list(self, meal_plan):
        """
        Generate a shopping list from the meal plan
        """
        ingredients = {}
        
        for day in meal_plan:
            for meal in day["meals"].values():
                if meal:
                    for ing in meal["ingredients"]:
                        ingredients[ing] = ingredients.get(ing, 0) + 1
                        
        return {"ingredients": ingredients}

if __name__ == "__main__":
    chatbot = MealPlanChatbot()
    chatbot.start_chat()