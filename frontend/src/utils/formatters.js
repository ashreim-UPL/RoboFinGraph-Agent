export const formatAgentRoleKey = (key) =>
  key.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
