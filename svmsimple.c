#include <stdio.h>
#include <lbfgs.h>
#include <stdlib.h>
#include <math.h>

/* structure for use by eval_svm */
typedef struct svm_institer_type{
	int n;
	int d;
	lbfgsfloatval_t lambda;
	lbfgsfloatval_t huberconst;	/* value of constant h */

	lbfgsfloatval_t **data;
	lbfgsfloatval_t *b;
}svminstiter;

/* structure for storing the parameters for the problem */
typedef struct __tproblem{
	lbfgsfloatval_t **train_data;
	lbfgsfloatval_t *train_labels;
	int n;
	int d;
	lbfgsfloatval_t lambda;
	lbfgsfloatval_t huberconst;
	lbfgsfloatval_t eps;
	lbfgsfloatval_t c;
}tproblem;

/* training routines */
int train_classifier_nonpriv(tproblem *p, lbfgsfloatval_t *c);
int train_classifier_outputperturbation(tproblem *p, lbfgsfloatval_t *c);
int train_classifier_objectiveperturbation(tproblem *p, lbfgsfloatval_t *c);
void train_all(tproblem *p, lbfgsfloatval_t **c, int *rvs);

/* generating random vector routines */
void generate_rand_vector(lbfgsfloatval_t *vector, int d, lbfgsfloatval_t epsilon,
    lbfgsfloatval_t scale);
lbfgsfloatval_t drawexp(lbfgsfloatval_t sigma);
lbfgsfloatval_t drawgauss();

/* matrix-vector routines */
void vecadd(int d, lbfgsfloatval_t *a, const lbfgsfloatval_t *b);
lbfgsfloatval_t vecdot(int d, const lbfgsfloatval_t *a, const lbfgsfloatval_t *b);
void scalarmult(int d, const lbfgsfloatval_t *a, lbfgsfloatval_t scalar, 
    lbfgsfloatval_t *b);
lbfgsfloatval_t	svmhlossgrad(const int d, const lbfgsfloatval_t h, const lbfgsfloatval_t *w, 
    const lbfgsfloatval_t *x, lbfgsfloatval_t *g);
lbfgsfloatval_t eval_svm(void *in, const lbfgsfloatval_t *x, lbfgsfloatval_t *grad,
    const int d, const lbfgsfloatval_t step);
void veccpy(int d, lbfgsfloatval_t *a, const lbfgsfloatval_t *b);

/* other routines */
int load_datafile(char *fname, tproblem *p);
void dump_vector(int d, lbfgsfloatval_t *v);
void outputclassifiers(int d, lbfgsfloatval_t **c, int *rv);


/*****implementations of functions****/
void dump_vector(int d, lbfgsfloatval_t *v){
  	int i;
	for(i=0; i<d; i++){
	  printf("%lf ", v[i]);
	}
	printf("\n");
}

int load_datafile(char *filename, tproblem *p){
  	/**routine to load data from a file generated by matlab **/
	FILE *fh;
	int i,j;
	lbfgsfloatval_t n,d,lam,eps, h;

	/*open the file and do some error checking*/
	fh = fopen(filename, "r");
	if (fh == NULL){
		fprintf(stderr, "Error opening file\n");
		return -1;
	}

	/*load the various sizes of the arrays and the two parameters*/
	fscanf(fh,"%lf", &n);
	fscanf(fh,"%lf", &d);
	fscanf(fh,"%lf", &lam);
	fscanf(fh,"%lf", &eps);
	fscanf(fh,"%lf", &h);

	p->n = (int) n;
	p->d = (int) d;
	p->lambda = lam;
	p->eps = eps;
	p->huberconst = h;
	p->c = 1/(2 * h);		/* value for svm */
	
	/**Allocate space for the training data */
	p->train_labels = (lbfgsfloatval_t *)malloc(p->n * sizeof(lbfgsfloatval_t));
	p->train_data = (lbfgsfloatval_t **) malloc(p->n * sizeof(lbfgsfloatval_t *));
	for (i=0; i<p->n; i++){
		p->train_data[i] = (lbfgsfloatval_t *) malloc(p->d *  sizeof(lbfgsfloatval_t));
	}
	
	/*load the data arrays*/	
	for (i=0; i<p->n; i++){
		for(j=0; j<p->d; j++){
			fscanf(fh, "%lf", &(p->train_data[i][j]));
		}
	}
	for (i=0; i<p->n; i++){
	  	fscanf(fh, "%lf", &(p->train_labels[i]));
	}
	/* at this point, p has the parameters of the problem, and the training data and labels */	
	
	fclose(fh);
	return 0;
}

void outputclassifiers(int d, lbfgsfloatval_t **c, int *rvs){
	int i, j;

	for(i=0; i<3; i++){
		for(j=0; j<d; j++){
			printf("%lf ", c[i][j]);
		}
		printf("  %d\n", rvs[i]);
	}
}

lbfgsfloatval_t vecdot(int d, const lbfgsfloatval_t *a, const lbfgsfloatval_t *b){
	int i;
	lbfgsfloatval_t dot = 0.0;

	for(i=0;i<d;i++){
		dot = dot + a[i] * b[i];
	}
	return dot;
}

void vecadd(int d, lbfgsfloatval_t *a, const lbfgsfloatval_t *b){
	/* add the vector b to the vector a*/
	int i;
	for(i=0;i<d; i++){
		a[i] = a[i] + b[i];
	}
}

void scalarmult(int d, const lbfgsfloatval_t *a, lbfgsfloatval_t scalar, 
    lbfgsfloatval_t *b){
	/*multiply vector a by scalar, and out result in b*/
	int i;
	for(i=0; i<d; i++){
		b[i] = scalar * a[i];
	}
}

/*update the gradient and loss*/
lbfgsfloatval_t	svmhlossgrad(const int d, const lbfgsfloatval_t huberconst, const lbfgsfloatval_t *w, 
    const lbfgsfloatval_t *x, lbfgsfloatval_t *g){
	int i;
	lbfgsfloatval_t dot;
	lbfgsfloatval_t hx;

	dot = vecdot(d, w,x);
	if (dot > 1.0 + huberconst){
		hx = 0;
		for (i=0;i<d;i++){
			g[i] = 0.0;
		}
	}else if (dot < 1.0 - huberconst){
		hx = 1 - dot;
		for(i=0;i<d;i++){
			g[i] = -x[i];
		}
	}else{
		hx = (1 + huberconst - dot) * (1 + huberconst - dot) / (4 * huberconst);
		for(i=0; i<d; i++){
			g[i] = - (1 + huberconst - dot) * x[i] / (2 * huberconst);
		}
	}
	return hx;
}

/**evaluate the function : svm function*/
lbfgsfloatval_t eval_svm(void *in, const lbfgsfloatval_t *x, lbfgsfloatval_t *grad, const int d, const lbfgsfloatval_t step){
	svminstiter *y = (svminstiter *)in;
	int i,j;

	/*temporarily holds the function and gradient values for each round*/
	lbfgsfloatval_t	fx;
	lbfgsfloatval_t *g = (lbfgsfloatval_t *)malloc(y->d * 
	    sizeof(lbfgsfloatval_t));	
	lbfgsfloatval_t onebyn = 1.0 / y->n;

	/*initialize*/
	fx = 0;
	for(i=0; i<y->d;i++){
		grad[i] = 0.0;
	}

	/*first,add the huber loss from all the data points*/
	for(i=0; i<y->n; i++){
		fx += onebyn * svmhlossgrad(y->d, y->huberconst, x, y->data[i], g);
		scalarmult(y->d, g, onebyn, g);
		vecadd(y->d, grad, g);		
	}

	/*now the effect of the regularization term and the b term*/
	fx = fx + 0.5 * y->lambda * vecdot(y->d, x,x) + 
	  onebyn * vecdot(y->d, y->b,x);
	
	scalarmult(y->d, x, y->lambda, g);
	vecadd(d, grad, g);
	scalarmult(y->d, y->b, onebyn, g);
	vecadd(d, grad, g);
			
	free(g);
	return fx;
}		

lbfgsfloatval_t drawgauss(){
  	/* uses the box-mueller transform */
	lbfgsfloatval_t t1, t2, x;

	t1 = (lbfgsfloatval_t) (rand() + 1.0)/(1.0 + RAND_MAX);
	t2 = (lbfgsfloatval_t) (rand() + 1.0)/(1.0 + RAND_MAX);
	x = sinf(2 * M_PI * t2) * sqrtf(-2 * logf(t1));
	return x;
}

lbfgsfloatval_t drawexp(lbfgsfloatval_t sigma){
  	lbfgsfloatval_t a, b;

	a = (lbfgsfloatval_t) (rand() + 1.0)/(1.0 + RAND_MAX);
	b = -logf(a) * sigma;
	return b;
}

/*given an epsilon, d, scale, generate a random vector from norm scale * Gamma(d,
 * 1/\epsilon), direction u a r */
void generate_rand_vector(lbfgsfloatval_t *vector, int d, lbfgsfloatval_t epsilon,
    lbfgsfloatval_t scale){
  	int i;
  	lbfgsfloatval_t norm = 0.0;
	lbfgsfloatval_t sum = 0.0;

  	for(i=0; i<d; i++){
  		norm = norm + scale * drawexp(1.0/epsilon);
  	}
  	for(i=0; i<d; i++){
  		vector[i] = (1.0/sqrtf((lbfgsfloatval_t)d)) * drawgauss();
		sum = sum + vector[i] * vector[i];
  	}
	sum = sqrtf(sum + 1e-10);
	for(i=0; i<d; i++){
	  	vector[i] = norm * vector[i] / sum;
	}
}


void train_all(tproblem *p, lbfgsfloatval_t **c, int *rvs){
	/* subroutine for training all the classifiers and collecting their return values */
	int i, j;
	tproblem q;

	q.n = p->n; 
	q.d = p->d;
	q.lambda = p->lambda;
	q.eps = p->eps;
	q.c = p->c;
	q.huberconst = p->huberconst;

	/* The only difference between p and q is that q multiplies all the training data vectors by
	   their labels. This is used by the svm function, which only needs the vectors y_i * x_i */
	q.train_labels = NULL;
	q.train_data = (lbfgsfloatval_t **)malloc( q.n * sizeof(lbfgsfloatval_t *));
	for(i=0; i<q.n; i++){
		q.train_data[i] = (lbfgsfloatval_t *) malloc( q.d * sizeof(lbfgsfloatval_t));
		for(j=0; j<q.d; j++){
			q.train_data[i][j] = p->train_data[i][j] * p->train_labels[i];
		}
	}
	
	rvs[0] = train_classifier_nonpriv(&q, c[0]);
	rvs[1] = train_classifier_outputperturbation(&q, c[1]);
	rvs[2] = train_classifier_objectiveperturbation(&q, c[2]);

	for(i=0; i<q.n; i++){
		free(q.train_data[i]);
	}
	free(q.train_data);
}

int train_classifier_nonpriv(tproblem *p, lbfgsfloatval_t *normal){
	/* Train a non-private logistic regression classifier */
	svminstiter l;
	int i;
	lbfgs_parameter_t param;
	lbfgsfloatval_t *xopt;
	lbfgsfloatval_t fx;
	int ret;

  	l.n = p->n;
	l.d = p->d;
	l.lambda = p->lambda;
	l.data = p->train_data;
	l.huberconst = p->huberconst;
	l.b = (lbfgsfloatval_t *)malloc(p->d * sizeof(lbfgsfloatval_t));
	for(i=0; i<p->d; i++){
	  l.b[i] = 0.0;
	}
	xopt = lbfgs_malloc(p->d);
	lbfgs_parameter_init(&param);
	
	ret = lbfgs(p->d, xopt, &fx, &eval_svm, NULL, (void *)&l, &param);
	for(i=0; i<p->d; i++){
	  normal[i] = xopt[i];
	}
	free(l.b);
	lbfgs_free(xopt);
	return ret;
}

int train_classifier_outputperturbation(tproblem *p, lbfgsfloatval_t *classifier){
	/* Train a regularized LR classifier by the sensitivity method */
	int i, ret;
	lbfgsfloatval_t scale;
	lbfgsfloatval_t *normal;

	/*first train a non-private LR classifier, then add noise */
	normal = (lbfgsfloatval_t *)malloc(p->d * sizeof(lbfgsfloatval_t));
	ret = train_classifier_nonpriv(p, normal);

	if (ret == 0){	/* optimization converged during training, so output perturbed classifier */
		scale = 2.0/( (lbfgsfloatval_t)p->n * p->lambda);
		generate_rand_vector(classifier, p->d, p->eps, scale);
		vecadd(p->d, classifier, normal);
		}
	free(normal);
	return ret;
}		

int train_classifier_objectiveperturbation(tproblem *p, lbfgsfloatval_t *classifier){
	/* Train a regularized SVM classifier by the objective perturbation method */ 
	svminstiter l;
	int i,j;
	lbfgs_parameter_t param;
	lbfgsfloatval_t *xopt;
	lbfgsfloatval_t fx;
	lbfgsfloatval_t epsp;
	lbfgsfloatval_t y;
	int limit;
	int count, ret;

  	l.n = p->n;
	l.d = p->d;
	l.lambda = p->lambda;
	l.data = p->train_data;
	l.huberconst = p->huberconst;	
	l.b = (lbfgsfloatval_t *)malloc(p->d * sizeof(lbfgsfloatval_t));
	xopt = lbfgs_malloc(p->d);
	lbfgs_parameter_init(&param);
		
	y = p->c / ( p->lambda * ((lbfgsfloatval_t) p->n));
	epsp = p->eps - logf(1.0 + 2 * y + y * y);
	if (epsp < 1e-4){
	  	fprintf(stderr, "Error: Cannot run algorithm for this lambda, epsilon value");
		return -1;
	}
	
	limit = 20;
	count = 0; 
	while(count < limit){
		count ++;
		generate_rand_vector(l.b, p->d, epsp, 2.0);
		ret = lbfgs(p->d, xopt, &fx, &eval_svm, NULL, (void *)&l, &param);
		if (ret >= 0){
		   for(j=0; j<p->d; j++){
	 	      classifier[j] = xopt[j];
		   }
		   break;
		}
	}
	free(l.b);
	lbfgs_free(xopt);
	
	if (count >= limit){
	  return -1;	/* tried 20 times, but could not get a successful optimization */
	}else{
	  return 0;	/*successful completion */
	}
}
	
void veccpy(int n, lbfgsfloatval_t *a, const lbfgsfloatval_t *b){
	int i;
	for(i=0; i<n; i++){
		a[i] = b[i];
	}
}
			   
int driver(char *dataf){
  	int r, i;
	tproblem p;
	lbfgsfloatval_t **classifiers;
	int retvals[3];

	/* first, load the data file to find parameters of the problem p*/
	if (load_datafile(dataf, &p) < 0){
	  fprintf(stderr, "Error loading data file\n");
	  return -1;
	}
	
	/* now allocate memory for the three classifiers, and train them */
	classifiers = (lbfgsfloatval_t **)malloc(sizeof(lbfgsfloatval_t *) * 3);
	for(i=0; i<3; i++){
		classifiers[i] = (lbfgsfloatval_t *) malloc(p.d * sizeof(lbfgsfloatval_t));
	}
	train_all(&p, classifiers, retvals);
			
	/* finally, output them to disk, along with the return values, which 
	indicate whether the optimization converged or not */
	outputclassifiers(p.d, classifiers, retvals);	

	/* free memory */
	free(classifiers[0]); free(classifiers[1]); free(classifiers[2]);
	free(classifiers);

	return 0;
}


int main(int argc, char *argv[]){
	int ret;
  	if (argc < 2){
	  fprintf(stdout, "Error: Usage: %s <data-file>\n", argv[0]);
	  return -1;
	}
  	
  	ret = driver(argv[1]);
	return ret;
}






