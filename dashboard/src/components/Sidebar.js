import React from 'react';
import { NavLink, useLocation } from 'react-router-dom';
import {
  HomeIcon,
  UserGroupIcon,
  ChartBarIcon,
  CogIcon,
  ArrowRightOnRectangleIcon,
} from '@heroicons/react/24/outline';
import { useAuth } from '../contexts/AuthContext';

const navigation = [
  { name: 'Dashboard', href: '/', icon: HomeIcon },
  { name: 'Patient Cohort', href: '/cohort', icon: UserGroupIcon },
  { name: 'Analytics', href: '/analytics', icon: ChartBarIcon },
];

const Sidebar = () => {
  const { user, logout } = useAuth();
  const location = useLocation();

  return (
    <div className="hidden lg:fixed lg:inset-y-0 lg:z-50 lg:flex lg:w-80 lg:flex-col">
      <div className="flex grow flex-col gap-y-6 overflow-y-auto bg-gradient-to-b from-slate-900 via-slate-800 to-slate-900 px-6 shadow-2xl border-r border-slate-700">
        <div className="flex h-20 shrink-0 items-center">
          <div className="flex items-center">
            <div className="flex-shrink-0">
              <div className="h-12 w-12 bg-gradient-to-br from-blue-500 to-indigo-600 rounded-xl flex items-center justify-center shadow-lg">
                <span className="text-white font-bo ld text-lg">E-ZC</span>
              </div>
            </div>
            <div className="ml-4">
              <h1 className="text-xl font-bold text-white">
                E-ZenithCare
              </h1>
              <p className="text-sm text-slate-300"></p>
            </div>
          </div>
        </div>
        
        <nav className="flex flex-1 flex-col">
          <ul className="flex flex-1 flex-col">
            <li className="flex-1">
              <ul className="-mx-2 space-y-2">
                {navigation.map((item) => (
                  <li key={item.name}>
                    <NavLink
                      to={item.href}
                      className={({ isActive }) =>
                        `group flex gap-x-4 rounded-xl p-3 text-sm leading-6 font-semibold transition-all duration-200 transform hover:scale-105 ${
                          isActive
                            ? 'bg-gradient-to-r from-blue-600 to-indigo-600 text-white shadow-lg'
                            : 'text-slate-300 hover:text-white hover:bg-slate-700/50'
                        }`
                      }
                    >
                      <item.icon
                        className="h-6 w-6 shrink-0"
                        aria-hidden="true"
                      />
                      {item.name}
                    </NavLink>
                  </li>
                ))}
              </ul>
            </li>
            
            <li className="mt-auto">
              <div className="border-t border-slate-700 pt-6">
                <div className="flex items-center px-3 py-4 bg-slate-800/50 rounded-xl mb-4">
                  <div className="flex-shrink-0">
                    <div className="h-10 w-10 bg-gradient-to-br from-blue-500 to-indigo-600 rounded-full flex items-center justify-center shadow-lg">
                      <span className="text-white font-bold text-sm">
                        {user?.name?.charAt(0) || 'D'}
                      </span>
                    </div>
                  </div>
                  <div className="ml-3 flex-1">
                    <p className="text-sm font-semibold text-white">
                      Dr. Sarah Johnson
                    </p>
                    <p className="text-xs text-slate-400 capitalize">
                      Clinician
                    </p>
                  </div>
                </div>
                
                <button
                  onClick={logout}
                  className="group flex w-full gap-x-3 rounded-xl p-3 text-sm leading-6 font-semibold text-slate-300 hover:text-white hover:bg-red-600/20 border border-slate-600 hover:border-red-500/50 transition-all duration-200 transform hover:scale-105"
                >
                  <ArrowRightOnRectangleIcon
                    className="h-5 w-5 shrink-0"
                    aria-hidden="true"
                  />
                  Sign out
                </button>
              </div>
            </li>
          </ul>
        </nav>
      </div>
    </div>
  );
};

export default Sidebar;
