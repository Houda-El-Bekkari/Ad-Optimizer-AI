import { Routes } from '@angular/router';

import { AppShell } from './layouts/app-shell/app-shell';
import { Accueil } from './pages/accueil/accueil';
import { AiAudit } from './pages/ai-audit/ai-audit';
import { AiAssistant } from './pages/ai-assistant/ai-assistant';
import { CampaignPlanner } from './pages/campaign-planner/campaign-planner';
import { Campaigns } from './pages/campaigns/campaigns';
import { ConnectPlatforms } from './pages/connect-platforms/connect-platforms';
import { Dashboard } from './pages/dashboard/dashboard';
import { DecisionInsights } from './pages/decision-insights/decision-insights';
import { Optimization } from './pages/optimization/optimization';
import { Reports } from './pages/reports/reports';
import { Settings } from './pages/settings/settings';
import { Signup } from './pages/signup/signup';
import { Login } from './pages/login/login';

export const routes: Routes = [
  {
    path: '',
    component: Accueil,
    pathMatch: 'full',
    title: 'Accueil - AdOptimizer AI',
  },
  {
  path: 'signup',
  component: Signup,
  title: 'Signup - AdOptimizer AI',
},
  {
    path: 'login',
    component: Login,
    title: 'Connexion - AdOptimizer AI',
  },
  
  {
    path: 'connect-platforms',
    component: ConnectPlatforms,
    title: 'Connexion plateformes - AdOptimizer AI',
  },
  {
    path: 'ai-audit',
    component: AiAudit,
    title: 'Audit intelligent - AdOptimizer AI',
  },
  {
    path: '',
    component: AppShell,
    children: [
      {
        path: 'dashboard',
        component: Dashboard,
        title: 'Overview - AdOptimizer AI',
      },
      {
        path: 'campaigns',
        component: Campaigns,
        title: 'Campaigns - AdOptimizer AI',
      },
      {
        path: 'optimization',
        component: Optimization,
        title: 'Optimization - AdOptimizer AI',
      },
      {
        path: 'campaign-planner',
        component: CampaignPlanner,
        title: 'Campaign Planner - AdOptimizer AI',
      },
      {
        path: 'decision-insights',
        component: DecisionInsights,
        title: 'Decision Insights - AdOptimizer AI',
      },
      {
        path: 'ai-assistant',
        component: AiAssistant,
        title: 'AI Assistant - AdOptimizer AI',
      },
      {
        path: 'reports',
        component: Reports,
        title: 'Reports - AdOptimizer AI',
      },
      {
        path: 'settings',
        component: Settings,
        title: 'Settings - AdOptimizer AI',
      },
    ],
  },
  {
    path: '**',
    redirectTo: '',
  },
];
