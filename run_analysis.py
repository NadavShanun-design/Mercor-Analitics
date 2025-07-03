# -*- coding: utf-8 -*-
"""
Job Offer Acceptance Likelihood Analysis
Runs the analysis on actual Mercor user data
"""

import pandas as pd
import numpy as np
import re
from typing import List, Dict, Tuple
import warnings
warnings.filterwarnings('ignore')

class JobAcceptancePredictor:
    def __init__(self):
        self.ml_engineer_skills = ['python', 'tensorflow', 'machine learning', 'data science', 'sql', 'ai', 'deep learning', 'ml']
        self.fullstack_skills = ['javascript', 'react', 'node.js', 'python', 'sql', 'html', 'css', 'web development', 'frontend', 'backend']
        
    def load_data(self, users_file: str) -> pd.DataFrame:
        """Load and preprocess user data"""
        print(f"Loading data from {users_file}...")
        users = pd.read_csv(users_file)
        
        print(f"Loaded {len(users)} users")
        print(f"Columns: {list(users.columns)}")
        
        # Clean and prepare data
        users['summary_clean'] = users['summary'].fillna('').str.lower()
        users['fullTimeSalary'] = pd.to_numeric(users['fullTimeSalary'], errors='coerce').fillna(0)
        users['partTimeSalary'] = pd.to_numeric(users['partTimeSalary'], errors='coerce').fillna(0)
        
        # Filter active users only
        active_users = users[users['isActive'] == True].copy()
        print(f"Active users: {len(active_users)}")
        
        return active_users
    
    def extract_skills(self, text: str, required_skills: List[str]) -> List[str]:
        """Extract skills from text using keyword matching"""
        if pd.isna(text) or text == '':
            return []
        
        found_skills = []
        for skill in required_skills:
            if skill in text.lower():
                found_skills.append(skill)
        
        return found_skills
    
    def calculate_skill_match(self, user_summary: str, required_skills: List[str]) -> float:
        """Calculate skill match score"""
        found_skills = self.extract_skills(user_summary, required_skills)
        if len(required_skills) == 0:
            return 0.0
        return len(found_skills) / len(required_skills)
    
    def calculate_experience_match(self, user_data: pd.Series, min_years: int) -> float:
        """Calculate experience match score"""
        summary = user_data.get('summary_clean', '')
        
        # Look for experience indicators
        experience_indicators = [
            r'(\d+)\s*years?\s*experience',
            r'(\d+)\s*years?\s*in',
            r'(\d+)\s*years?\s*of',
            r'senior',
            r'lead',
            r'manager',
            r'(\d+)\+?\s*years'
        ]
        
        for pattern in experience_indicators:
            matches = re.findall(pattern, summary, re.IGNORECASE)
            if matches:
                try:
                    years = int(matches[0])
                    return 1.0 if years >= min_years else 0.0
                except:
                    continue
        
        # Check availability indicators
        if user_data.get('fullTime', False) or user_data.get('partTime', False):
            return 0.5  # Some experience indicated
        
        return 0.0
    
    def calculate_pay_attractiveness(self, user_data: pd.Series, job_pay: float) -> float:
        """Calculate pay attractiveness score"""
        fulltime_salary = user_data.get('fullTimeSalary', 0)
        parttime_salary = user_data.get('partTimeSalary', 0)
        
        # Convert annual salary to hourly rate (assuming 2080 hours/year)
        if fulltime_salary > 0:
            current_hourly = fulltime_salary / 2080
        elif parttime_salary > 0:
            current_hourly = parttime_salary
        else:
            return 0.5  # No salary data, assume neutral
        
        # Compare job pay to current expectations
        if job_pay > current_hourly * 1.2:
            return 1.0  # Job pays significantly more
        elif job_pay >= current_hourly * 0.8:
            return 0.7  # Job pays similar or slightly more
        else:
            return 0.3  # Job pays less
    
    def calculate_location_compatibility(self, user_location: str, job_remote: bool = True) -> float:
        """Calculate location compatibility score"""
        if pd.isna(user_location) or user_location == '':
            return 0.5  # Unknown location, assume neutral
        
        # For remote jobs, check if user is in a compatible time zone
        us_locations = ['united states', 'canada', 'mexico']
        eu_locations = ['united kingdom', 'germany', 'france', 'spain', 'italy', 'netherlands', 'sweden']
        
        location_lower = user_location.lower()
        
        if any(us in location_lower for us in us_locations):
            return 1.0  # US timezone, highly compatible
        elif any(eu in location_lower for eu in eu_locations):
            return 0.8  # EU timezone, good compatibility
        else:
            return 0.6  # Other locations, moderate compatibility
    
    def calculate_confidence_score(self, user_data: pd.Series, job_config: Dict) -> float:
        """Calculate overall confidence score for job acceptance"""
        required_skills = job_config['skills']
        min_experience = job_config['min_experience']
        job_pay = job_config['pay_rate']
        job_remote = job_config.get('remote', True)
        
        # Calculate individual scores
        skill_match = self.calculate_skill_match(user_data['summary_clean'], required_skills)
        experience_match = self.calculate_experience_match(user_data, min_experience)
        pay_attractiveness = self.calculate_pay_attractiveness(user_data, job_pay)
        location_compatibility = self.calculate_location_compatibility(user_data.get('location', ''), job_remote)
        
        # Weighted average
        confidence_score = (
            skill_match * 0.4 +
            experience_match * 0.3 +
            pay_attractiveness * 0.2 +
            location_compatibility * 0.1
        )
        
        return min(max(confidence_score, 0), 1)  # Clamp between 0 and 1
    
    def predict_acceptance_likelihood(self, users_file: str) -> pd.DataFrame:
        """Main function to predict job acceptance likelihood for all users"""
        users = self.load_data(users_file)
        
        # Define job configurations
        ml_engineer_job = {
            'skills': self.ml_engineer_skills,
            'min_experience': 3,
            'pay_rate': 75,
            'remote': True
        }
        
        fullstack_job = {
            'skills': self.fullstack_skills,
            'min_experience': 2,
            'pay_rate': 60,
            'remote': True
        }
        
        print("Calculating ML Engineer scores...")
        # Calculate scores for each job
        users['ml_engineer_score'] = users.apply(
            lambda row: self.calculate_confidence_score(row, ml_engineer_job), axis=1
        )
        
        print("Calculating Full-Stack Developer scores...")
        users['fullstack_score'] = users.apply(
            lambda row: self.calculate_confidence_score(row, fullstack_job), axis=1
        )
        
        # Add overall score (average of both jobs)
        users['overall_acceptance_score'] = (users['ml_engineer_score'] + users['fullstack_score']) / 2
        
        return users

def main():
    """Main function to run the analysis"""
    print("=== Job Offer Acceptance Likelihood Analysis ===\n")
    
    # Initialize predictor
    predictor = JobAcceptancePredictor()
    
    # Run analysis
    try:
        results = predictor.predict_acceptance_likelihood('sql_6_2025-02-22T1512.csv')
        
        # Display summary statistics
        print("\n=== SUMMARY STATISTICS ===")
        print(f"Total users analyzed: {len(results)}")
        print(f"Average ML Engineer score: {results['ml_engineer_score'].mean():.3f}")
        print(f"Average Full-Stack score: {results['fullstack_score'].mean():.3f}")
        print(f"Average overall score: {results['overall_acceptance_score'].mean():.3f}")
        
        # High-confidence candidates
        high_confidence = results[results['overall_acceptance_score'] > 0.7]
        print(f"High-confidence candidates (>0.7): {len(high_confidence)} ({len(high_confidence)/len(results)*100:.1f}%)")
        
        # Display top candidates
        print("\n=== TOP 10 CANDIDATES ===")
        top_candidates = results.nlargest(10, 'overall_acceptance_score')[
            ['userId', 'name', 'ml_engineer_score', 'fullstack_score', 'overall_acceptance_score', 'location']
        ]
        
        print(top_candidates.to_string(index=False))
        
        # Skill distribution analysis
        print("\n=== SKILL DISTRIBUTION ===")
        python_users = results[results['summary_clean'].str.contains('python', na=False)]
        ml_users = results[results['summary_clean'].str.contains('machine learning|ml|ai', na=False)]
        js_users = results[results['summary_clean'].str.contains('javascript|react|node', na=False)]
        
        print(f"Users with Python skills: {len(python_users)} ({len(python_users)/len(results)*100:.1f}%)")
        print(f"Users with ML/AI skills: {len(ml_users)} ({len(ml_users)/len(results)*100:.1f}%)")
        print(f"Users with JavaScript/React skills: {len(js_users)} ({len(js_users)/len(results)*100:.1f}%)")
        
        # Geographic distribution
        print("\n=== GEOGRAPHIC DISTRIBUTION ===")
        us_users = results[results['location'].str.contains('United States|Canada', na=False)]
        eu_users = results[results['location'].str.contains('United Kingdom|Germany|France', na=False)]
        
        print(f"US/Canada users: {len(us_users)} ({len(us_users)/len(results)*100:.1f}%)")
        print(f"EU users: {len(eu_users)} ({len(eu_users)/len(results)*100:.1f}%)")
        
        # Save results
        output_file = 'job_acceptance_results.csv'
        results.to_csv(output_file, index=False)
        print(f"\nResults saved to: {output_file}")
        
    except FileNotFoundError:
        print("Error: Could not find the CSV file. Please ensure 'sql_6_2025-02-22T1512.csv' is in the current directory.")
    except Exception as e:
        print(f"Error running analysis: {str(e)}")

if __name__ == "__main__":
    main() 