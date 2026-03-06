import numpy as np
import daqp
USE_QPSOLVERS = True

if USE_QPSOLVERS:
    from qpsolvers import solve_qp, Problem, solve_problem
else:
    import cvxopt


class cbf_si():
    def __init__(self, P = None, q = None):
        self.reset_cbf()
        self.warm_start = False
        self.u1_prev = None  # Previous control input for warm start

    def reset_cbf(self):
        # initialize G and h, Then fill it afterwards
        self.constraint_G = None
        self.constraint_h = None
        self.cbf_values = None

    def __set_constraint(self, G_mat, h_mat):
        if self.constraint_G is None:
            self.constraint_G = G_mat
            self.constraint_h = h_mat
        else:
            self.constraint_G = np.append(self.constraint_G, G_mat, axis=0)
            self.constraint_h = np.append(self.constraint_h, h_mat, axis=0)

    def add_computed_constraint(self, G_mat, h_mat): self.__set_constraint(G_mat, h_mat)

    def compute_safe_controller(self, u_nom,speed_limit, P = None, q = None):

        if (P is None) and (q is None):
            if u_nom is None:
                print('WARNING: No nominal control input provided!!!!')
            P= 2*np.eye(2)  # Identity matrix for quadratic cost
            q =  -2*u_nom

        if self.constraint_G is not None:
            if USE_QPSOLVERS:
                def_ublb = np.inf
                self._var_num = 2 # to allow additional variable if needed
                lb = np.ones(self._var_num)*(-def_ublb)
                ub = np.ones(self._var_num)*(def_ublb)

                if speed_limit is not None:
                    self.add_velocity_bound(speed_limit-1e-7)  # Add velocity bound constraints
                    #array_limit = np.ones(2)* (speed_limit-1E-7)
                    #lb[:2], ub[:2] = -array_limit, array_limit

                # SCALING PROCESS
                G_mat = self.constraint_G.copy()
                h_mat = self.constraint_h.copy()
                # IMPLEMENTATION OF Control Barrier Function

                opt_tolerance = 1e-8 

                qp_problem = Problem(P, q, G_mat, h_mat, lb = lb, ub = ub,)
                
                #qp_problem.check_primal_feasibility()
                qp_problem.check_constraints()
                # qp_problem.cond()
                # solution = solve_problem(qp_problem, solver="daqp")
                
                if self.warm_start:
                    #qp_problem.x0= self.u1_prev
                    solution = solve_problem(qp_problem, solver="daqp",initvals=self.u1_prev, dual_tol=opt_tolerance, primal_tol=opt_tolerance)
                    #solution = solve_problem(qp_problem, solver="daqp",initvals=self.u1_prev)
                else:
                    solution = solve_problem(qp_problem, solver="daqp", dual_tol=opt_tolerance, primal_tol=opt_tolerance)
                    #solution = solve_problem(qp_problem, solver="daqp")

                # solution = solve_problem(qp_problem, solver="clarabel")
                # solution = solve_problem(qp_problem, solver="clarabel", 
                #          tol_feas=1e-9, tol_gap_abs=1e-9, tol_gap_rel=0)
                # solution = solve_problem(qp_problem, solver="quadprog")
                #


                if solution is None:
                    print( 'WARNING QP SOLVER [no solution] stopping instead')
                    u_star = np.array([0., 0., 0.])

                else:
                    sol = solution.x
                    if sol is None or not solution.is_optimal(opt_tolerance):
                        
                        #print('P',P,'q', q,'G', G_mat,'h', h_mat,'lb' , lb, 'ub',ub)
                        if self.u1_prev is None:
                            print( 'WARNING QP SOLVER [not optimal or no x] stopping instead')
                            u_star = np.array([0., 0., 0.])
                        else:
                            print( 'WARNING QP SOLVER [not optimal or no x] using prev solution instead' )
                            u_star = self.u1_prev.copy()

                    else:
                        u_star = np.array([sol[0], sol[1], 0])
                        self.warm_start = True
                        self.u1_prev = np.array([sol[0],  sol[1]]) # warm start with previous control input.copy()

            else:
                #print('WARNING QP SOLVER [cvxopt] using cvxopt instead, this is deprecated, use daqp instead!!!!!!!!')
                # IMPLEMENTATION OF Control Barrier Function
                # Minimization
                P_mat = cvxopt.matrix( P.astype(np.double), tc='d')
                q_mat = cvxopt.matrix( q.astype(np.double), tc='d')
                # Resize the G and H into appropriate matrix for optimization
                G_mat = cvxopt.matrix( self.constraint_G.astype(np.double), tc='d') 
                h_mat = cvxopt.matrix( self.constraint_h.astype(np.double), tc='d')

                # Solving Optimization
                cvxopt.solvers.options['show_progress'] = False
                sol = cvxopt.solvers.qp(P_mat, q_mat, G_mat, h_mat, verbose=False)

                if sol['status'] == 'optimal':
                    # Get solution + converting from cvxopt base matrix to numpy array
                    u_star = np.array([sol['x'][0], sol['x'][1]])
                else: 
                    print( 'WARNING QP SOLVER' + ' status: ' + sol['status'] + ' --> use nominal instead' )
                    u_star = u_nom.copy()
        else: # No constraints imposed
            u_star = u_nom.copy()

        #ret_h = {} # TODO

        # TODO: output h value and label
        return u_star


    # ADDITION OF CONSTRAINTS
    # -----------------------------------------------------------------------------------------------------------
    def add_avoid_static_circle(self, pos, obs, ds, gamma=10, power=3):
        # h = norm2( pos - obs )^2 - norm2(ds)^2 > 0
        vect = pos - obs
        h_func = np.power(np.linalg.norm(vect), 2) - np.power(ds, 2)
        # -(dh/dpos)^T u < gamma(h)
        self.__set_constraint(-2*vect.reshape((1,3)), gamma*np.power(h_func, power).reshape((1,1)))

        return h_func


    def add_maintain_distance_with_epsilon(self, pos, obs, ds, epsilon, gamma=10, power=3):
        vect = pos - obs
        # h = norm2( ds + epsilon )^2 - norm2( pos - obs )^2 > 0
        h_func_l = np.power((ds+epsilon), 2) - np.power(np.linalg.norm(vect), 2)
        # -(dh/dpos)^T u < gamma(h)
        self.__set_constraint(2*vect.reshape((1,3)), gamma*np.power(h_func_l, power).reshape((1,1)))

        # h = norm2( pos - obs )^2 - norm2( ds - epsilon )^2 > 0
        h_func_u = np.power(np.linalg.norm(vect), 2) - np.power((ds-epsilon), 2)
        # -(dh/dpos)^T u < gamma(h)
        self.__set_constraint(-2*vect.reshape((1,3)), gamma*np.power(h_func_u, power).reshape((1,1)))

        return h_func_l, h_func_u


    def add_avoid_static_ellipse(self, pos, obs, theta, major_l, minor_l, gamma=10, power=3):
        # h = norm2( ellipse*[pos - obs] )^2 - 1 > 0
        theta = theta if np.ndim(theta) == 0 else theta.item()
        # TODO: assert a should be larger than b (length of major axis vs minor axis)
        vect = pos - obs # compute vector towards pos from centroid
        # rotate vector by -theta (counter the ellipse angle)
        # then skew the field due to ellipse major and minor axis
        # the resulting vector should be grater than 1
        # i.e. T(skew)*R(-theta)*vec --> then compute L2norm square
        ellipse = np.array([[2./major_l, 0, 0], [0, 2./minor_l, 0], [0, 0, 1]]) \
            @ np.array([[np.cos(-theta), -np.sin(-theta), 0], [np.sin(-theta), np.cos(-theta), 0], [0, 0, 1]], dtype=object)
        h_func = np.power(np.linalg.norm( ellipse @ vect.T ), 2) - 1
        # -(dh/dpos)^T u < gamma(h)
        # -(2 vect^T ellipse^T ellipse) u < gamma(h)
        G = -2*vect @ ( ellipse.T @ ellipse )
        self.__set_constraint( G.reshape((1,3)), gamma*np.power(h_func, power).reshape((1,1)) )

        return h_func


    def add_velocity_bound(self, vel_limit):
        G = np.vstack((np.eye(2), -np.eye(2)))
        h = np.ones([4, 1]) * vel_limit
        self.__set_constraint( G, h )

    # TODO: add area with boundary