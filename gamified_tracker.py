import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple
import json

class GamifiedLearningTracker:
    """Comprehensive gamified learning progress tracker for federated learning"""
    
    def __init__(self):
        self.achievements = self._initialize_achievements()
        self.progress_milestones = self._initialize_milestones()
        self.leaderboard_data = {}
        self.user_stats = {}
        
    def _initialize_achievements(self) -> Dict[str, Dict]:
        """Initialize achievement system with badges and rewards"""
        return {
            "first_training": {
                "name": "🚀 First Steps",
                "description": "Complete your first federated learning training round",
                "icon": "🚀",
                "points": 100,
                "category": "beginner",
                "unlocked": False
            },
            "accuracy_milestone": {
                "name": "🎯 Accuracy Master",
                "description": "Achieve 85% accuracy or higher",
                "icon": "🎯",
                "points": 250,
                "category": "performance",
                "unlocked": False
            },
            "convergence_expert": {
                "name": "⚡ Convergence Expert",
                "description": "Achieve model convergence in under 10 rounds",
                "icon": "⚡",
                "points": 300,
                "category": "efficiency",
                "unlocked": False
            },
            "privacy_guardian": {
                "name": "🛡️ Privacy Guardian",
                "description": "Complete training with differential privacy enabled",
                "icon": "🛡️",
                "points": 200,
                "category": "security",
                "unlocked": False
            },
            "marathon_trainer": {
                "name": "🏃 Marathon Trainer",
                "description": "Complete 20 or more training rounds",
                "icon": "🏃",
                "points": 400,
                "category": "endurance",
                "unlocked": False
            },
            "collaboration_champion": {
                "name": "🤝 Collaboration Champion",
                "description": "Successfully train with 5+ medical facilities",
                "icon": "🤝",
                "points": 350,
                "category": "collaboration",
                "unlocked": False
            },
            "data_scientist": {
                "name": "📊 Data Scientist",
                "description": "Analyze all performance metrics and visualizations",
                "icon": "📊",
                "points": 150,
                "category": "analysis",
                "unlocked": False
            },
            "innovation_pioneer": {
                "name": "💡 Innovation Pioneer",
                "description": "Use advanced aggregation algorithms (FedProx)",
                "icon": "💡",
                "points": 275,
                "category": "innovation",
                "unlocked": False
            },
            "quality_assurance": {
                "name": "✅ Quality Assurance",
                "description": "Maintain F1-score above 0.80 for 5 consecutive rounds",
                "icon": "✅",
                "points": 225,
                "category": "quality",
                "unlocked": False
            },
            "network_architect": {
                "name": "🌐 Network Architect",
                "description": "Explore all graph visualization types",
                "icon": "🌐",
                "points": 175,
                "category": "exploration",
                "unlocked": False
            }
        }
    
    def _initialize_milestones(self) -> List[Dict]:
        """Initialize progress milestones with rewards"""
        return [
            {"level": 1, "points": 100, "title": "Novice Researcher", "reward": "🌟 Welcome Badge"},
            {"level": 2, "points": 300, "title": "Junior Data Scientist", "reward": "📈 Progress Tracker"},
            {"level": 3, "points": 600, "title": "ML Practitioner", "reward": "🔬 Advanced Analytics"},
            {"level": 4, "points": 1000, "title": "Federated Learning Expert", "reward": "🏆 Expert Status"},
            {"level": 5, "points": 1500, "title": "AI Research Leader", "reward": "👑 Leadership Crown"},
            {"level": 6, "points": 2200, "title": "Innovation Master", "reward": "💎 Master Achievement"},
            {"level": 7, "points": 3000, "title": "Grand Master", "reward": "🎖️ Grand Master Medal"}
        ]
    
    def update_progress(self, training_data: Dict[str, Any]) -> Dict[str, Any]:
        """Update progress based on training results and unlock achievements"""
        updates = {
            "new_achievements": [],
            "level_up": False,
            "current_level": self.get_current_level(),
            "total_points": self.get_total_points()
        }
        
        # Check for achievement unlocks
        if training_data.get('rounds_completed', 0) >= 1:
            if not self.achievements["first_training"]["unlocked"]:
                self.achievements["first_training"]["unlocked"] = True
                updates["new_achievements"].append("first_training")
        
        if training_data.get('final_accuracy', 0) >= 0.85:
            if not self.achievements["accuracy_milestone"]["unlocked"]:
                self.achievements["accuracy_milestone"]["unlocked"] = True
                updates["new_achievements"].append("accuracy_milestone")
        
        if training_data.get('converged', False) and training_data.get('rounds_completed', 999) <= 10:
            if not self.achievements["convergence_expert"]["unlocked"]:
                self.achievements["convergence_expert"]["unlocked"] = True
                updates["new_achievements"].append("convergence_expert")
        
        if training_data.get('privacy_enabled', False):
            if not self.achievements["privacy_guardian"]["unlocked"]:
                self.achievements["privacy_guardian"]["unlocked"] = True
                updates["new_achievements"].append("privacy_guardian")
        
        if training_data.get('rounds_completed', 0) >= 20:
            if not self.achievements["marathon_trainer"]["unlocked"]:
                self.achievements["marathon_trainer"]["unlocked"] = True
                updates["new_achievements"].append("marathon_trainer")
        
        if training_data.get('num_clients', 0) >= 5:
            if not self.achievements["collaboration_champion"]["unlocked"]:
                self.achievements["collaboration_champion"]["unlocked"] = True
                updates["new_achievements"].append("collaboration_champion")
        
        if training_data.get('aggregation_algorithm') == 'FedProx':
            if not self.achievements["innovation_pioneer"]["unlocked"]:
                self.achievements["innovation_pioneer"]["unlocked"] = True
                updates["new_achievements"].append("innovation_pioneer")
        
        # Check for level up
        old_level = self.get_current_level()
        new_total_points = self.get_total_points()
        new_level = self.get_level_from_points(new_total_points)
        
        if new_level > old_level:
            updates["level_up"] = True
            updates["new_level"] = new_level
        
        updates["current_level"] = new_level
        updates["total_points"] = new_total_points
        
        return updates
    
    def get_total_points(self) -> int:
        """Calculate total points from unlocked achievements"""
        return sum(
            achievement["points"] 
            for achievement in self.achievements.values() 
            if achievement["unlocked"]
        )
    
    def get_current_level(self) -> int:
        """Get current level based on total points"""
        total_points = self.get_total_points()
        return self.get_level_from_points(total_points)
    
    def get_level_from_points(self, points: int) -> int:
        """Convert points to level"""
        for milestone in reversed(self.progress_milestones):
            if points >= milestone["points"]:
                return milestone["level"]
        return 0
    
    def get_next_milestone(self) -> Dict[str, Any]:
        """Get information about the next milestone"""
        current_points = self.get_total_points()
        
        for milestone in self.progress_milestones:
            if current_points < milestone["points"]:
                return {
                    "milestone": milestone,
                    "points_needed": milestone["points"] - current_points,
                    "progress_percentage": (current_points / milestone["points"]) * 100
                }
        
        # Already at max level
        return {
            "milestone": {"title": "Grand Master", "points": current_points},
            "points_needed": 0,
            "progress_percentage": 100
        }
    
    def create_progress_dashboard(self) -> None:
        """Create comprehensive gamified progress dashboard"""
        st.header("🎮 Gamified Learning Progress")
        
        # Current status overview
        self._create_status_overview()
        
        # Achievement gallery
        self._create_achievement_gallery()
        
        # Progress visualization
        self._create_progress_visualization()
        
        # Leaderboard
        self._create_leaderboard()
        
        # Statistics and insights
        self._create_statistics_panel()
    
    def _create_status_overview(self) -> None:
        """Create current status overview panel"""
        st.subheader("📊 Current Status")
        
        col1, col2, col3, col4 = st.columns(4)
        
        current_level = self.get_current_level()
        total_points = self.get_total_points()
        achievements_unlocked = len([a for a in self.achievements.values() if a["unlocked"]])
        total_achievements = len(self.achievements)
        
        with col1:
            st.metric("Current Level", f"Level {current_level}")
            if current_level > 0:
                current_title = self.progress_milestones[current_level - 1]["title"]
                st.caption(f"🏆 {current_title}")
        
        with col2:
            st.metric("Total Points", f"{total_points:,}")
            
        with col3:
            st.metric("Achievements", f"{achievements_unlocked}/{total_achievements}")
            completion_rate = (achievements_unlocked / total_achievements) * 100
            st.caption(f"🎯 {completion_rate:.1f}% Complete")
        
        with col4:
            next_milestone = self.get_next_milestone()
            if next_milestone["points_needed"] > 0:
                st.metric("Next Level", f"{next_milestone['points_needed']} pts")
                st.caption(f"📈 {next_milestone['progress_percentage']:.1f}%")
            else:
                st.metric("Status", "MAX LEVEL")
                st.caption("🎖️ Grand Master")
    
    def _create_achievement_gallery(self) -> None:
        """Create achievement gallery with unlocked and locked badges"""
        st.subheader("🏆 Achievement Gallery")
        
        # Group achievements by category
        categories = {}
        for key, achievement in self.achievements.items():
            category = achievement["category"]
            if category not in categories:
                categories[category] = []
            categories[category].append((key, achievement))
        
        # Display achievements by category
        for category, achievements in categories.items():
            with st.expander(f"📁 {category.title()} Achievements", expanded=True):
                cols = st.columns(min(len(achievements), 3))
                
                for idx, (key, achievement) in enumerate(achievements):
                    with cols[idx % 3]:
                        if achievement["unlocked"]:
                            st.success(f"""
                            **{achievement['icon']} {achievement['name']}** ✅
                            
                            {achievement['description']}
                            
                            **Points:** {achievement['points']} 🎖️
                            """)
                        else:
                            st.info(f"""
                            **🔒 {achievement['name']}** 
                            
                            {achievement['description']}
                            
                            **Reward:** {achievement['points']} points
                            """)
    
    def _create_progress_visualization(self) -> None:
        """Create progress visualization charts"""
        st.subheader("📈 Progress Visualization")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Level progress bar
            next_milestone = self.get_next_milestone()
            current_level = self.get_current_level()
            
            fig_progress = go.Figure()
            
            # Progress bar
            fig_progress.add_trace(go.Bar(
                x=[next_milestone['progress_percentage']],
                y=['Level Progress'],
                orientation='h',
                marker=dict(color='lightblue'),
                text=f"{next_milestone['progress_percentage']:.1f}%",
                textposition='inside',
                name='Progress'
            ))
            
            fig_progress.update_layout(
                title=f"Progress to Level {current_level + 1}",
                xaxis=dict(range=[0, 100], title="Completion %"),
                height=200,
                showlegend=False
            )
            
            st.plotly_chart(fig_progress, use_container_width=True)
        
        with col2:
            # Achievement completion by category
            category_stats = {}
            for achievement in self.achievements.values():
                category = achievement["category"]
                if category not in category_stats:
                    category_stats[category] = {"total": 0, "unlocked": 0}
                category_stats[category]["total"] += 1
                if achievement["unlocked"]:
                    category_stats[category]["unlocked"] += 1
            
            categories = list(category_stats.keys())
            completion_rates = [
                (stats["unlocked"] / stats["total"]) * 100 
                for stats in category_stats.values()
            ]
            
            fig_categories = go.Figure(data=[
                go.Bar(
                    x=categories,
                    y=completion_rates,
                    marker=dict(color=px.colors.qualitative.Set3[:len(categories)]),
                    text=[f"{rate:.0f}%" for rate in completion_rates],
                    textposition='outside'
                )
            ])
            
            fig_categories.update_layout(
                title="Achievement Completion by Category",
                yaxis=dict(range=[0, 100], title="Completion %"),
                height=300
            )
            
            st.plotly_chart(fig_categories, use_container_width=True)
    
    def _create_leaderboard(self) -> None:
        """Create leaderboard with top performers"""
        st.subheader("🏅 Global Leaderboard")
        
        # Simulate leaderboard data for demonstration
        leaderboard_data = [
            {"rank": 1, "user": "AI_Researcher_Pro", "level": 7, "points": 3200, "achievements": 10},
            {"rank": 2, "user": "ML_Master_2024", "level": 6, "points": 2800, "achievements": 9},
            {"rank": 3, "user": "Fed_Learning_Expert", "level": 6, "points": 2400, "achievements": 8},
            {"rank": 4, "user": "Data_Science_Ninja", "level": 5, "points": 1800, "achievements": 7},
            {"rank": 5, "user": "Current_User", "level": self.get_current_level(), "points": self.get_total_points(), "achievements": len([a for a in self.achievements.values() if a["unlocked"]])},
        ]
        
        # Sort by points
        leaderboard_data.sort(key=lambda x: x["points"], reverse=True)
        
        # Update ranks
        for i, entry in enumerate(leaderboard_data):
            entry["rank"] = i + 1
        
        # Display leaderboard
        leaderboard_df = pd.DataFrame(leaderboard_data)
        
        # Style the current user row
        def highlight_current_user(row):
            if row["user"] == "Current_User":
                return ['background-color: lightblue'] * len(row)
            return [''] * len(row)
        
        styled_df = leaderboard_df.style.apply(highlight_current_user, axis=1)
        st.dataframe(styled_df, use_container_width=True)
        
        # Leaderboard insights
        current_rank = next(entry["rank"] for entry in leaderboard_data if entry["user"] == "Current_User")
        st.info(f"🎯 You are currently ranked #{current_rank} globally!")
    
    def _create_statistics_panel(self) -> None:
        """Create detailed statistics and insights panel"""
        st.subheader("📊 Detailed Statistics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**🎯 Achievement Statistics**")
            
            # Points by category
            category_points = {}
            for achievement in self.achievements.values():
                if achievement["unlocked"]:
                    category = achievement["category"]
                    if category not in category_points:
                        category_points[category] = 0
                    category_points[category] += achievement["points"]
            
            if category_points:
                fig_pie = go.Figure(data=[go.Pie(
                    labels=list(category_points.keys()),
                    values=list(category_points.values()),
                    hole=0.3
                )])
                
                fig_pie.update_layout(
                    title="Points Distribution by Category",
                    height=300
                )
                
                st.plotly_chart(fig_pie, use_container_width=True)
            else:
                st.info("Start training to see your achievement distribution!")
        
        with col2:
            st.markdown("**📈 Progress Timeline**")
            
            # Simulate progress timeline
            if any(achievement["unlocked"] for achievement in self.achievements.values()):
                timeline_data = []
                for key, achievement in self.achievements.items():
                    if achievement["unlocked"]:
                        timeline_data.append({
                            "achievement": achievement["name"],
                            "points": achievement["points"],
                            "date": datetime.now() - timedelta(days=np.random.randint(0, 30))
                        })
                
                if timeline_data:
                    timeline_df = pd.DataFrame(timeline_data)
                    timeline_df = timeline_df.sort_values("date")
                    
                    fig_timeline = px.line(
                        timeline_df, 
                        x="date", 
                        y="points",
                        title="Points Earned Over Time",
                        markers=True
                    )
                    
                    fig_timeline.update_layout(height=300)
                    st.plotly_chart(fig_timeline, use_container_width=True)
            else:
                st.info("Complete achievements to see your progress timeline!")
    
    def show_achievement_popup(self, achievement_key: str) -> None:
        """Show achievement unlock popup"""
        if achievement_key in self.achievements:
            achievement = self.achievements[achievement_key]
            
            st.balloons()
            st.success(f"""
            🎉 **ACHIEVEMENT UNLOCKED!** 🎉
            
            {achievement['icon']} **{achievement['name']}**
            
            {achievement['description']}
            
            **Reward:** {achievement['points']} points! 🎖️
            """)
    
    def export_progress_report(self) -> Dict[str, Any]:
        """Export comprehensive progress report"""
        return {
            "current_level": self.get_current_level(),
            "total_points": self.get_total_points(),
            "achievements_unlocked": [
                {
                    "name": achievement["name"],
                    "points": achievement["points"],
                    "category": achievement["category"]
                }
                for achievement in self.achievements.values()
                if achievement["unlocked"]
            ],
            "progress_summary": {
                "achievements_completed": len([a for a in self.achievements.values() if a["unlocked"]]),
                "total_achievements": len(self.achievements),
                "completion_percentage": (len([a for a in self.achievements.values() if a["unlocked"]]) / len(self.achievements)) * 100
            },
            "next_milestone": self.get_next_milestone(),
            "export_timestamp": datetime.now().isoformat()
        }