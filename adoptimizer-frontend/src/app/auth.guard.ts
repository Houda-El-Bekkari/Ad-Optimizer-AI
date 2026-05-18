import { inject } from '@angular/core';
import { CanActivateFn, Router } from '@angular/router';

import { AuthSession } from './services/auth-session';

export const authGuard: CanActivateFn = () => {
  const session = inject(AuthSession);
  const router = inject(Router);

  if (session.isAuthenticated()) {
    return true;
  }

  return router.createUrlTree(['/login']);
};
