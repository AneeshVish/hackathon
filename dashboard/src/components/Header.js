import React from 'react';
import { BellIcon } from '@heroicons/react/24/outline';
import { useAuth } from '../contexts/AuthContext';

const Header = ({ searchTerm, setSearchTerm }) => {
  const { user } = useAuth();

  return (
    <div className="sticky top-0 z-40 flex h-16 shrink-0 items-center justify-between border-b border-gray-200 bg-white px-4 shadow-sm sm:px-6 lg:px-8">
      {/* Left side - Title */}
      <div className="flex items-center">
        <h1 className="text-xl font-semibold text-gray-900">E-ZenithCare Dashboard</h1>
      </div>
      
      {/* Right side - Navigation and Actions */}
      <div className="flex items-center gap-x-4">
        {/* Search bar - kept for future use but currently empty */}
        <div className="relative flex-1 max-w-md">
          {/* Search input can be added here later if needed */}
        </div>
        
        {/* Notification and User Profile */}
        <div className="flex items-center gap-x-4">
          <button
            type="button"
            className="relative -m-2.5 p-2.5 text-gray-400 hover:text-gray-500"
            title="Notifications"
          >
            <span className="sr-only">View notifications</span>
            <BellIcon className="h-6 w-6" aria-hidden="true" />
            <span className="absolute top-0 right-0 h-2 w-2 rounded-full bg-red-500"></span>
          </button>
          
          {/* User Profile */}
          <div className="flex items-center">
            <div className="h-8 w-8 rounded-full bg-blue-100 flex items-center justify-center">
              <span className="text-sm font-medium text-blue-700">
                {user?.username?.charAt(0).toUpperCase() || 'U'}
              </span>
            </div>
            <div className="ml-3">
              <p className="text-sm font-medium text-gray-700">{user?.name || 'User'}</p>
              <p className="text-xs text-gray-500">{user?.role || 'Clinician'}</p>
            </div>
          </div>

          <div
            className="hidden lg:block lg:h-6 lg:w-px lg:bg-gray-200"
            aria-hidden="true"
          />
        </div>
      </div>
    </div>
  );
};

export default Header;
